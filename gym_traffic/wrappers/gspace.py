import gym
from gym_traffic.spaces.gspace import GSpace
import numpy as np

# Translates gym envs with Box obs, Discrete actions to use GSpaces.

class GSpaceWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.observation_space = GSpace(env.observation_space.high)
    self.action_space = GSpace(env.action_space.n)

  def _step(self, action):
    obs, reward, done, info = self.env.step(np.asscalar(action))
    return np.reshape(np.array(obs), self.observation_space.shape), reward, done, info

  def _reset(self):
    return np.reshape(np.array(self.env.reset()), self.observation_space.shape)
