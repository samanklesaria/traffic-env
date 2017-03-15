import gym
import tensorflow as tf
from gym_traffic.wrappers.gspace import GSpaceWrapper
from gym_traffic.spaces.gspace import GSpace
import gym_traffic.algorithms.a3c as a3c
import numpy as np
from gym_traffic.algorithms.util import FLAGS

# Hack for testing training strategies using multidimensional rewards
class FakeVecWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.reward_size = 3 * env.reward_size
    self.observation_space = env.observation_space.replicated(4)
    self.action_space = env.action_space.replicated(3)

  def _step(self, action):
    obs, reward, done, info = self.env.step(action[0])
    return np.broadcast_to(obs, self.observation_space.shape), \
        np.broadcast_to(reward, (3, self.env.reward_size)), done, info

  def _reset(self):
    obs = self.env.reset()
    return np.broadcast_to(obs, self.observation_space.shape)

# Hack for embedding 1d observation space into hxw format expected
def make_env():
  w = GSpaceWrapper(gym.make('CartPole-v0'))
  old_limit = w.observation_space.limit
  w.observation_space = GSpace(np.expand_dims(np.expand_dims(w.observation_space.limit, 0), 0))
  return w

def main(_):
  FLAGS.learning_rate = 0.01
  FLAGS.episode_len = 600
  FLAGS.gamma = 0.99
  FLAGS.summary_rate = 10
  FLAGS.batch_size = 30
  FLAGS.lam = 1
  FLAGS.print_discounted = False
  globals()[FLAGS.trainer].run(make_env)

if __name__ == '__main__':
  tf.app.run()

