import gym
import tensorflow as tf
from gym_traffic.wrappers.gspace import GSpaceWrapper
import gym_traffic.algorithms.a3c as a3c
import numpy as np
from gym_traffic.algorithms.util import FLAGS
import origa3c

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

def main(_):
  FLAGS.learning_rate = 0.01
  FLAGS.episode_len = 600
  FLAGS.gamma = 0.99
  FLAGS.lam = 1
  globals()[FLAGS.trainer].run(lambda **_: GSpaceWrapper(gym.make('CartPole-v0')))
  # globals()[FLAGS.trainer].run(lambda **_: gym.make('CartPole-v0'))

if __name__ == '__main__':
  tf.app.run()

