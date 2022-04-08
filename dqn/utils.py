from collections import deque
import einops
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    FireResetEnv,
    EpisodicLifeEnv,
    MaxAndSkipEnv,
    ClipRewardEnv,
    WarpFrame,
)
import torch


ATARI_ENVS = ["Breakout", "Pong", "SpaceInvaders", "StarGunner"]


def preprocess_env(env, episodic_life=True):
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = ClipRewardEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, n_frames=4)
    return env


def batchify(state, add_channel_dim=False):
    batch_obs = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    if add_channel_dim:
        batch_obs = einops.rearrange(
            batch_obs, "b i j c -> b c i j"
        )  # batch, channel, row, column
    return batch_obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames
        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to np.ndarray before being passed to the model.
        :param frames: ([int] or [float]) environment frames
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
