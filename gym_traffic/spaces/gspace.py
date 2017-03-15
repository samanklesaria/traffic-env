import gym
import numpy as np

# Dealing with spaces generically sucks. Monomorphism makes everything easier.

class GSpace(gym.Space):
  def __init__(self, *args, **kwargs):
    self.limit = np.array(*args, **kwargs)
    self.shape = self.limit.shape
    self.size = self.limit.size

  def contains(self, x):
    xarr = np.array(x)
    return xarr.shape == self.limit.shape and \
      np.issubdtype(xarr.dtype,self.limit.dtype) and (x < self.limit).all()

  def sample(self):
    # Admittedly this isn't a uniform sample unless limits are uniform
    # LET's do a proper uniform sample.
    return np.random.randint(np.min(self.limit), size=self.limit.shape, dtype=self.limit.dtype)

  def empty(self): return np.empty_like(self.limit)

  def to_action(self, a):
    return np.reshape(np.array(a), self.limit.shape).astype(self.limit.dtype)

  def replicated(self, n):
    return GSpace(np.broadcast_to(self.limit, (n, *self.limit.shape)))
