import collections
import re
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

# `ogbench` and `utils.datasets` (which pulls in jax + flax) are imported inside
# the OGBench branch of `make_env_and_datasets` rather than here -- they are the
# only things in this module that need them, and importing them at module scope
# made `import envs.env_utils` require OGBench even for an env family that has
# nothing to do with it. A MetaWorld-only environment has no ogbench installed
# and used to fail at import time, before it ever reached `make_env_and_datasets`.
# Every other branch below already imports its dependencies this way.


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_extra_train_envs(env_name, count, seed=0):
    """Build `count` ADDITIONAL, independent copies of the online training env.

    `make_env_and_datasets` already returns one train env; this supplies the rest
    when a run collects from several envs per step (`num_train_envs` in
    main_cw2.py). They are plain, separate env objects stepped in sequence rather
    than a gymnasium vector env: the online loop keeps a per-env action-chunk
    queue and writes each env into its own replay-buffer segment, so it needs the
    envs individually, and BoxPushing's mujoco step is cheap next to the agent
    update that follows it.

    Only the env families that a multi-env config actually uses are implemented;
    anything else raises rather than silently running with one env.
    """
    if count <= 0:
        return []
    if env_name.startswith('fancy/'):
        from envs import box_pushing_utils

        return [box_pushing_utils.make_box_pushing_env(env_name, seed=seed + i) for i in range(count)]
    raise NotImplementedError(
        f'num_train_envs > 1 is not implemented for env_name={env_name!r}. '
        'Add a factory for its family in envs/env_utils.py::make_extra_train_envs.'
    )


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5, seed=0):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        seed: Env-side seed. Only the BoxPushing branch honours it; the other
            families keep the seeding they had before it was added, so runs that
            are in flight are not silently re-randomised.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        import ogbench

        from utils.datasets import Dataset

        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif env_name.startswith("lift") or env_name.startswith("can") or env_name.startswith("square") or \
        env_name.startswith("transport") or env_name.startswith("tool_hang"):
        # RoboMimic.
        from envs import robomimic_utils

        env = robomimic_utils.make_env(env_name, seed=0)
        eval_env = robomimic_utils.make_env(env_name, seed=42)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = robomimic_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif env_name.startswith('metaworld/'):
        # MetaWorld. No offline dataset exists for it -- online-only (offline_steps=0).
        from envs import metaworld_utils

        env, eval_env, train_dataset, val_dataset = metaworld_utils.make_metaworld_env_and_datasets(env_name)
    elif env_name.startswith('fancy/'):
        # fancy_gym BoxPushing. Like MetaWorld it ships no offline dataset, so it
        # is online-only (offline_steps=0).
        from envs import box_pushing_utils

        env, eval_env, train_dataset, val_dataset = box_pushing_utils.make_box_pushing_env_and_datasets(
            env_name, seed=seed)
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        if train_dataset is not None:
            train_dataset = train_dataset.copy(
                add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
