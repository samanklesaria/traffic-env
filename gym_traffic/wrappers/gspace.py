import gym
from gym_traffic.spaces.gspace import GSpace
from gym.spaces import Discrete, Box
import numpy as np

# Translates gym envs with Box obs, Discrete actions to use GSpaces.

class GSpaceWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    lim = np.float32(np.min(env.observation_space.high))
    self.observation_space = GSpace(env.observation_space.shape, lim)
    self.action_space = GSpace([1], np.int32(env.action_space.n))

  def _step(self, action):
    obs, reward, done, info = self.env.step(np.asscalar(action))
    return np.reshape(obs, self.observation_space.shape), np.array([reward]), done, info

  def _reset(self):
    return np.reshape(np.array(self.env.reset()), self.observation_space.shape)


class UnGSpaceWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.action_gspace = self.action_space
    self.action_space = Discrete(self.action_gspace.size)
    self.observation_gspace = self.observation_space
    self.observation_space = Box(0, self.action_gspace.limit, shape=self.observation_gspace.shape)
  
  def _step(self, action):
    real_action =  np.unravel_index(action, self.action_gspace.shape)
    obs, reward, done, info = self.env.step(real_action)
    return obs, np.mean(reward), done, info
