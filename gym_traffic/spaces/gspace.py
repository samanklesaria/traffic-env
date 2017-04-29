import gym
import numpy as np

class GSpace(gym.Space):
  def __init__(self, shape, l):
    assert type(shape) == list
    self.limit = l
    self.shape = shape
    self.size = np.prod(shape)

  def contains(self, x):
    return x.shape == self.shape

  def sample(self):
    return np.random.randint(self.limit, size=self.shape, dtype=self.limit.dtype)

  def empty(self): return np.empty(self.shape, dtype=self.limit.dtype)

  def to_action(self, a):
    return np.reshape(a, self.shape).astype(self.limit.dtype)

  def replicated(self, n):
    return GSpace([n]+self.shape, self.limit)
