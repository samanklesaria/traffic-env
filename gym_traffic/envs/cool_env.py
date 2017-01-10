import gym
from gym import spaces
import numpy as np

# Weight rewards per intersection
def localize(a, weight=3):
    n = a.shape[0]
    fullcounts = np.tile(a, (n,1))
    weighted = (np.diag(a) * (weight - 1) + fullcounts) / weight
    return np.sum(weighted, axis=1)

# A one step game- try and add something to the input to make the differences between successive
# elements small. The weights should get to -b * I for some b.
class ContinuousCool(gym.Env):
  metadata = {'render.modes': []}
  observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=3)
  action_space = observation_space
  
  def _step(self, action):
    self.state += action
    padded = np.pad(self.state, 1, 'edge')
    reward = -np.mean(np.maximum(2 * padded[1:-1] - padded[:-2] - padded[2:], 0))
    return action, reward, True, None
  
  def _reset(self):
    self.state = observation_space.sample()
    return self.state

# A discrete version of the same game
class DiscreteCool(ContinuousCool):
  action_space = spaces.MultiDiscrete([[-1, 1]] * 3)
  observation_space = spaces.Box(low=-1., high=1, shape=3)

