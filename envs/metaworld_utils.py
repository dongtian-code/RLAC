"""MetaWorld environment support.

Unlike every other environment family in this repo, MetaWorld has no bundled
offline dataset -- callers must run purely online (`offline_steps=0`).

MetaWorld 3.x registers its own gymnasium entry points as a side effect of
`import metaworld`; "Meta-World/goal_observable" is the single-task,
goal-in-the-observation variant this repo trains on, and it takes the concrete
task through an `env_name` kwarg.

This used to go through `fancy_gym`, which registered "metaworld/<task>-v2" ids
itself. That dependency is gone: fancy_gym targets gymnasium 0.29 (it subclasses
the `EnvCompatibility` wrapper that gymnasium 1.0 removed) and pins
mujoco==2.3.3, neither of which is compatible with the gymnasium 1.1 / mujoco
3.3 stack the rest of this repo needs.
"""
import re

import gymnasium
import metaworld  # noqa: F401 -- registers the "Meta-World/..." gymnasium ids

from envs.env_utils import EpisodeMonitor

# The gymnasium id MetaWorld 3.x registers for a single task whose goal is part
# of the observation. The task itself is passed as `env_name`.
_GOAL_OBSERVABLE_ID = 'Meta-World/goal_observable'


def is_metaworld_env(env_name):
    return env_name.startswith('metaworld/')


def _metaworld_task_name(env_name):
    """Map this repo's "metaworld/<task>-v2" ids onto MetaWorld 3.x's "<task>-v3".

    The "-v2" spelling predates the MetaWorld 3.x upgrade and is still what the
    configs use, so both suffixes are accepted and normalised to the v3 task
    names in `metaworld.ALL_V3_ENVIRONMENTS`.
    """
    task = re.sub(r'-v\d+$', '', env_name.split('/', 1)[1]) + '-v3'
    if task not in metaworld.ALL_V3_ENVIRONMENTS:
        raise ValueError(
            f'Unknown MetaWorld task {task!r} (from env_name {env_name!r}). '
            f'Expected one of the {len(metaworld.ALL_V3_ENVIRONMENTS)} tasks in '
            'metaworld.ALL_V3_ENVIRONMENTS.'
        )
    return task


def _unfreeze_task(env):
    """Restore MetaWorld's per-reset task distribution on a goal-observable env.

    MetaWorld's goal-observable classes are constructed *frozen*: the generated
    __init__ (metaworld/env_dict.py::_create_observable_goal_envs) does one
    throwaway reset and then sets `_freeze_rand_vec = True`, after which
    `SawyerXYZEnv._get_state_rand_vec()` returns the cached `_last_rand_vec` on
    every subsequent reset. Left alone the env replays ONE fixed object pose +
    goal forever, which means training solves a single instance rather than the
    distribution the benchmark is defined over, and evaluating a deterministic
    policy over N episodes yields N identical rollouts -- so a success *rate*
    collapses to 0.0 or 1.0.

    Unfreezing restores the draw from `_random_reset_space`; `seeded_rand_vec`
    makes that draw use the env's seeded `np_random` rather than the global
    numpy RNG, keeping runs reproducible per seed (the class __init__ already
    called `env.seed(seed)`).

    These are the two lines fancy_gym applied in
    `fancy_gym/meta/metaworld_adapter.py::make_metaworld`, i.e. what the
    published MetaWorld baselines were run through; dropping them when fancy_gym
    was removed silently changed the benchmark. Kept in step with the sibling
    SimbaV2 repo (`scale_rl/envs/metaworld.py`).
    """
    unwrapped = env.unwrapped
    unwrapped._freeze_rand_vec = False
    unwrapped.seeded_rand_vec = True
    return env


def make_metaworld_env_and_datasets(env_name, seed=0):
    """Make a MetaWorld env pair. Returns (env, eval_env, None, None) -- the
    dataset slots are always None since there is no offline data for MetaWorld.
    """
    task = _metaworld_task_name(env_name)
    # disable_env_checker: MetaWorld declares loose observation-space bounds, so
    # gymnasium's passive checker warns on every reset/step ("obs ... is not
    # within the observation space") and pays for a space check per step. Both
    # are pure overhead here.
    # Only `env_name` and `seed` may be passed: MetaWorld registers this id with
    # `entry_point=lambda env_name, seed: ...` -- no **kwargs -- so any other
    # keyword (render_mode, reward_function_version, ...) is a TypeError from
    # gymnasium's env creator. The reward version therefore comes from the env
    # class default, which is "v2" on all 50 tasks, i.e. the benchmark reward.
    env = gymnasium.make(
        _GOAL_OBSERVABLE_ID, env_name=task, seed=seed, disable_env_checker=True)
    eval_env = gymnasium.make(
        _GOAL_OBSERVABLE_ID, env_name=task, seed=seed + 1000, disable_env_checker=True)
    _unfreeze_task(env)
    _unfreeze_task(eval_env)
    env = EpisodeMonitor(env)
    eval_env = EpisodeMonitor(eval_env)
    # MetaWorld's own seeding is unreliable, so this is best-effort, not a
    # reproducibility guarantee.
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)
    return env, eval_env, None, None
