"""MetaWorld environment support.

Unlike every other environment family in this repo, MetaWorld has no bundled
offline dataset -- callers must run purely online (`offline_steps=0`).

MetaWorld gym ids ("metaworld/<task>-v2") are registered by `fancy_gym` as a
side effect of importing it (see `fancy_gym.meta.metaworld_adapter`), not by
the `metaworld` package itself, so `import fancy_gym` has to happen before
`gymnasium.make(...)` is called.
"""
import fancy_gym  # noqa: F401 -- import for its gymnasium.register(...) side effects
import gymnasium

from envs.env_utils import EpisodeMonitor


def is_metaworld_env(env_name):
    return env_name.startswith('metaworld/')


def make_metaworld_env_and_datasets(env_name, seed=0):
    """Make a MetaWorld env pair. Returns (env, eval_env, None, None) -- the
    dataset slots are always None since there is no offline data for MetaWorld.
    """
    env = gymnasium.make(env_name)
    eval_env = gymnasium.make(env_name)
    env = EpisodeMonitor(env)
    eval_env = EpisodeMonitor(eval_env)
    # MetaWorld's own seeding is unreliable (fancy_gym's own adapter tests skip
    # determinism checks for it), so this is best-effort, not a reproducibility
    # guarantee.
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)
    return env, eval_env, None, None
