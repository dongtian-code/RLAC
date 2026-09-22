"""fancy_gym BoxPushing environment support.

Like MetaWorld, BoxPushing has no bundled offline dataset -- callers must run
purely online (`offline_steps=0`).

Why this does NOT `import fancy_gym`
------------------------------------
`fancy_gym/__init__.py` pulls in `fancy_gym.utils.make_env_helpers`, which
imports `fancy_gym.utils.env_compatibility`, which subclasses
`gymnasium.wrappers.EnvCompatibility` -- removed in gymnasium 1.0. On the
gymnasium 1.1 / mujoco 3.3 stack this repo needs (requirements.txt), `import
fancy_gym` therefore dies with an AttributeError before it registers a single
env id, which is why `envs/metaworld_utils.py` dropped fancy_gym in the first
place. Installing the package is still fine as long as pip is told not to
resolve its dependencies (it pins `mujoco==2.3.3`, which would downgrade mujoco
out from under metaworld, dm_control and ogbench):

    pip install --no-deps "git+https://github.com/DongTian95/fancy_gymnasium.git"

or, on the cluster, the shared `$DEPS_DIR/fancy_gymnasium` checkout that
`conda_env.sh` installs with `RLAC_WITH_BOXPUSHING=1`.

The env module itself is clean: `box_pushing_env.py` only uses the modern
gymnasium API (`MujocoEnv(..., observation_space=...)`, 5-tuple `step`), so it
runs unmodified on gymnasium 1.1. `_load_box_pushing_module` therefore loads
that one file (plus the pure-numpy `box_pushing_utils` it imports) straight off
disk under its canonical dotted name, with stub parent packages in `sys.modules`
so the intra-package import resolves without any `__init__.py` ever executing.
The 18 MB of Franka meshes under `assets/` stay in the installed package rather
than being vendored into this repo.

Episode structure (fancy_gym's own registration, reproduced here):
  * 100 steps, enforced by the env itself -- `BoxPushingEnvBase.step` sets
    `terminated = is_success` and `truncated = not is_success` once
    `_steps >= MAX_EPISODE_STEPS_BOX_PUSHING`. No `TimeLimit` wrapper is added:
    it would truncate on exactly the same step.
  * 29-d observation whose FIRST element is the step counter, so the fixed
    horizon is part of the state and the 100-step cut-off stays Markovian.
  * 7-d normalised joint-torque actions in [-1, 1].
  * `is_success` in `info` (box within 5 cm and 0.5 rad of the target), non-zero
    only on the final step.
"""
import hashlib
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import types

import gymnasium

from envs.env_utils import EpisodeMonitor

_ENV_PREFIX = 'fancy/'

# Mirrors fancy_gym/envs/__init__.py: the "RandomInit" ids are the same reward
# classes constructed with `random_init=True`, which redraws the box's starting
# pose every reset instead of always starting it at (0.4, 0.3). The goal pose is
# resampled every reset either way.
_BOX_PUSHING_ENVS = {
    'fancy/BoxPushingDense-v0': ('BoxPushingDense', {}),
    'fancy/BoxPushingTemporalSparse-v0': ('BoxPushingTemporalSparse', {}),
    'fancy/BoxPushingTemporalSpatialSparse-v0': ('BoxPushingTemporalSpatialSparse', {}),
    'fancy/BoxPushingRandomInitDense-v0': ('BoxPushingDense', {'random_init': True}),
    'fancy/BoxPushingRandomInitTemporalSparse-v0': (
        'BoxPushingTemporalSparse', {'random_init': True}),
    'fancy/BoxPushingRandomInitTemporalSpatialSparse-v0': (
        'BoxPushingTemporalSpatialSparse', {'random_init': True}),
}

# Parent packages of box_pushing_env, innermost last. Each gets a stub module
# with a `__path__` so the real `__init__.py` files never run.
_STUB_PACKAGES = (
    ('fancy_gym', ()),
    ('fancy_gym.envs', ('envs',)),
    ('fancy_gym.envs.mujoco', ('envs', 'mujoco')),
    ('fancy_gym.envs.mujoco.box_pushing', ('envs', 'mujoco', 'box_pushing')),
)

_MODULE_NAME = 'fancy_gym.envs.mujoco.box_pushing.box_pushing_env'

# fancy_gym's model XMLs were written for MuJoCo 2.x. MuJoCo 3 dropped
# <option collision="...">, and its compiler rejects unknown attributes outright:
#     ValueError: XML Error: Schema violation: unrecognized attribute: 'collision'
# "all" was 2.x's default (check every geom pair), so deleting the attribute is a
# no-op for the simulation -- it is the only thing standing between this env and
# the mujoco 3.3 stack this repo runs on. Patching happens in a cache directory
# rather than in the installed package: fancy_gym is a shared, read-only checkout
# that the sibling dt_rl repo runs against on mujoco 2.3.3, and this keeps working
# against an unmodified upstream clone.
_XML_FIXUPS = (
    (re.compile(r'\s+collision="[^"]*"'), ''),
)


def _patch_xml(text):
    for pattern, replacement in _XML_FIXUPS:
        text = pattern.sub(replacement, text)
    return text


def _asset_cache_root():
    override = os.environ.get('FANCY_GYM_ASSET_CACHE_DIR')
    if override:
        return override
    return os.path.join(
        os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
        'fancy_gym_mujoco3_assets',
    )


def _mujoco3_assets(pkg_dir):
    """Return an assets directory whose XMLs MuJoCo 3 will load.

    The package's own directory is returned unchanged when nothing needs
    patching, so a fancy_gym that has since been fixed upstream costs nothing.
    Otherwise the (small, 35-file) tree is copied once into a content-addressed
    cache directory; concurrent reps racing to build it is fine, the loser just
    reuses the winner's copy.
    """
    src = os.path.join(pkg_dir, 'assets')
    patched = {}
    for root, _, files in os.walk(src):
        for name in sorted(files):
            if not name.endswith('.xml'):
                continue
            path = os.path.join(root, name)
            with open(path) as f:
                text = f.read()
            new_text = _patch_xml(text)
            if new_text != text:
                patched[os.path.relpath(path, src)] = new_text
    if not patched:
        return src

    digest = hashlib.sha1(
        '\0'.join(f'{k}\0{v}' for k, v in sorted(patched.items())).encode('utf-8')
    ).hexdigest()[:12]
    cache_root = _asset_cache_root()
    dst = os.path.join(cache_root, f'box_pushing_{digest}')
    if os.path.isdir(dst):
        return dst

    os.makedirs(cache_root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f'box_pushing_{digest}.', dir=cache_root)
    try:
        staged = os.path.join(tmp, 'assets')
        shutil.copytree(src, staged)
        for rel, text in patched.items():
            with open(os.path.join(staged, rel), 'w') as f:
                f.write(text)
        try:
            os.rename(staged, dst)
        except OSError:
            # Another rep won the race (or the directory appeared meanwhile).
            if not os.path.isdir(dst):
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'[box_pushing] MuJoCo-3 asset cache: {dst}', flush=True)
    return dst


class _Mujoco3Assets:
    """Load the model from `_MUJOCO3_XML` instead of the package's own copy.

    `BoxPushingEnvBase.__init__` hardcodes its `model_path`, so the redirect has
    to happen after `MujocoEnv.__init__` has resolved `self.fullpath` and before
    it compiles the model -- which is exactly `_initialize_simulation`.
    """

    _MUJOCO3_XML = None

    def _initialize_simulation(self):
        if self._MUJOCO3_XML is not None:
            self.fullpath = self._MUJOCO3_XML
        return super()._initialize_simulation()


def is_box_pushing_env(env_name):
    return env_name.startswith(_ENV_PREFIX)


def _exec_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before executing: `box_pushing_env` imports `box_pushing_utils`
    # by its dotted name, and the import machinery looks in sys.modules first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_box_pushing_module():
    """Return fancy_gym's `box_pushing_env` module, importing it in isolation."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]

    # find_spec locates the package without executing its __init__.py.
    spec = importlib.util.find_spec('fancy_gym')
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            'fancy_gym is not installed, so the BoxPushing envs are unavailable. '
            'Install it WITHOUT its dependencies (it pins mujoco==2.3.3, which '
            'would downgrade this environment):\n'
            '    pip install --no-deps "git+https://github.com/DongTian95/fancy_gymnasium.git"'
        )
    root = list(spec.submodule_search_locations)[0]
    pkg_dir = os.path.join(root, 'envs', 'mujoco', 'box_pushing')
    if not os.path.isdir(pkg_dir):
        raise ImportError(
            f'Found fancy_gym at {root!r}, but no box_pushing package at {pkg_dir!r}.'
        )

    added = []
    try:
        for name, parts in _STUB_PACKAGES:
            if name in sys.modules:
                continue
            module = types.ModuleType(name)
            module.__path__ = [os.path.join(root, *parts)]
            module.__package__ = name
            sys.modules[name] = module
            added.append(name)
            parent, _, leaf = name.rpartition('.')
            if parent:
                setattr(sys.modules[parent], leaf, module)
        # box_pushing_utils first: box_pushing_env imports names from it.
        for leaf in ('box_pushing_utils', 'box_pushing_env'):
            name = f'fancy_gym.envs.mujoco.box_pushing.{leaf}'
            if name not in sys.modules:
                added.append(name)
                _exec_module(name, os.path.join(pkg_dir, f'{leaf}.py'))
    except Exception:
        # Leave sys.modules as it was found, so a later genuine `import
        # fancy_gym` is not silently served a half-built stub.
        for name in reversed(added):
            sys.modules.pop(name, None)
        raise

    return sys.modules[_MODULE_NAME]


class SuccessInfo(gymnasium.Wrapper):
    """Expose BoxPushing's `is_success` flag under the `success` key as well.

    MetaWorld reports `success`, BoxPushing reports `is_success`, and
    `evaluation.py` simply averages whatever the final step's `info` holds. Both
    keys are kept so a BoxPushing run logs `eval/success` like every MetaWorld
    run does and the two land on the same axis in the comparison plots.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if 'is_success' in info:
            info['success'] = float(info['is_success'])
        return observation, reward, terminated, truncated, info


_ENV_CLASSES = {}


def _env_class(class_name):
    """fancy_gym's env class, subclassed to read the MuJoCo-3-patched model."""
    if class_name in _ENV_CLASSES:
        return _ENV_CLASSES[class_name]
    module = _load_box_pushing_module()
    base = getattr(module, class_name)
    assets = _mujoco3_assets(os.path.dirname(module.__file__))
    if assets == os.path.join(os.path.dirname(module.__file__), 'assets'):
        cls = base                         # nothing needed patching
    else:
        cls = type(
            f'Mujoco3{class_name}', (_Mujoco3Assets, base),
            {'_MUJOCO3_XML': os.path.join(assets, 'box_pushing.xml')},
        )
    _ENV_CLASSES[class_name] = cls
    return cls


def make_box_pushing_env(env_name, seed=0):
    """Make a single BoxPushing env."""
    if env_name not in _BOX_PUSHING_ENVS:
        raise ValueError(
            f'Unknown BoxPushing env {env_name!r}. Expected one of: '
            f'{sorted(_BOX_PUSHING_ENVS)}.'
        )
    class_name, kwargs = _BOX_PUSHING_ENVS[env_name]
    env = _env_class(class_name)(**kwargs)
    env = SuccessInfo(env)
    env = EpisodeMonitor(env)
    # Seeds `np_random`, which draws the box and goal poses in `sample_context`.
    env.reset(seed=seed)
    return env


def make_box_pushing_env_and_datasets(env_name, seed=0):
    """Make a BoxPushing env pair. Returns (env, eval_env, None, None) -- the
    dataset slots are always None since there is no offline data for it.
    """
    env = make_box_pushing_env(env_name, seed=seed)
    # Far enough above the train seeds that it cannot collide with the extra
    # train envs (seed + 1, seed + 2, ...) of this or any other repetition.
    eval_env = make_box_pushing_env(env_name, seed=seed + 10000)
    return env, eval_env, None, None
