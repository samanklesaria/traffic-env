import gym
import tensorflow as tf
from gym_traffic.wrappers.gspace import GSpaceWrapper
from gym_traffic.spaces.gspace import GSpace
import gym_traffic.algorithms.qrnn as qrnn
import gym_traffic.algorithms.qlearn as qlearn
import gym_traffic.algorithms.a3c as a3c
import numpy as np
from args import parse_flags, update_flags, FLAGS

# Hack for testing training strategies using multidimensional rewards
# class FakeVecWrapper(gym.Wrapper):
#   def __init__(self, env):
#     super().__init__(env)
#     self.reward_size = 3 * env.reward_size
#     self.observation_space = env.observation_space.replicated(4)
#     self.action_space = env.action_space.replicated(3)
#
#   def _step(self, action):
#     obs, reward, done, info = self.env.step(action[0])
#     return np.broadcast_to(obs, self.observation_space.shape), \
#         np.broadcast_to(reward, (3, self.env.reward_size)), done, info
#
#   def _reset(self):
#     obs = self.env.reset()
#     return np.broadcast_to(obs, self.observation_space.shape)

def make_env():
  return GSpaceWrapper(gym.make('CartPole-v0'))

if __name__ == '__main__':
  parse_flags()
  update_flags(
    learning_rate = 0.01,
    episode_len = 800,
    gamma = 0.99,
    summary_rate = 20,
    save_rate = 10000,
    train_rate = 1,
    batch_size = 100,
    target_update_rate = 5,
    annealing_episodes = 1000,
    buffer_size = 50,
    lam = 1,
    start_eps = 0.2,
    min_eps = 0.01,
    reward_printing = "sum",
    trace_size = 1,
    validate_rate = 20)
  globals()[FLAGS.trainer].run(make_env)
