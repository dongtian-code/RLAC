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


def make_metaworld_env_and_datasets(env_name, seed=0):
    """Make a MetaWorld env pair. Returns (env, eval_env, None, None) -- the
    dataset slots are always None since there is no offline data for MetaWorld.
    """
    task = _metaworld_task_name(env_name)
    # disable_env_checker: MetaWorld declares loose observation-space bounds, so
    # gymnasium's passive checker warns on every reset/step ("obs ... is not
    # within the observation space") and pays for a space check per step. Both
    # are pure overhead here.
    env = gymnasium.make(
        _GOAL_OBSERVABLE_ID, env_name=task, seed=seed, disable_env_checker=True)
    eval_env = gymnasium.make(
        _GOAL_OBSERVABLE_ID, env_name=task, seed=seed + 1000, disable_env_checker=True)
    env = EpisodeMonitor(env)
    eval_env = EpisodeMonitor(eval_env)
    # MetaWorld's own seeding is unreliable, so this is best-effort, not a
    # reproducibility guarantee.
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)
    return env, eval_env, None, None
