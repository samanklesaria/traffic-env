import gym
import gym_traffic
from gym_traffic.spaces.gspace import GSpace
from gym_traffic.envs.roadgraph import GridRoad
from gym_traffic.wrappers.warmup import WarmupWrapper
import tensorflow as tf
import numpy as np
import json
import gym_traffic.algorithms.a3c as a3c
import gym_traffic.algorithms.random as random
import gym_traffic.algorithms.const0 as const0
import gym_traffic.algorithms.const1 as const1
import gym_traffic.algorithms.util
import gym_traffic.algorithms.fixed as fixed

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('episode_secs', 600, 'Secs per episode')
flags.DEFINE_integer('light_secs', 5, 'Seconds per light')
flags.DEFINE_float('warmup_lights', 2, 'Number of lights to choose randomly')
flags.DEFINE_integer('local_weight', 2, 'Weight to give local elements')
flags.DEFINE_boolean('squish_rewards', True, "Should we take an average of vector rewards?")

EPS = 1e-8

# Repeats the same action, summing 'passed' observations, taking the last of others
def Repeater(repeat_count):
  class Repeater(gym.Wrapper):
    def __init__(self, env):
      super(Repeater, self).__init__(env)
      self.r = self.unwrapped.graph.roads
      self.i = self.unwrapped.graph.intersections
      self.observation_space = GSpace(np.ones(self.r + self.i, dtype=np.float32))
    def _step(self, action):
      done = False
      total_reward = 0
      total_obs = np.zeros(self.r + self.i, dtype=np.float32)
      for _ in range(repeat_count):
        obs, reward, done, info = self.env.step(action)
        total_obs[:self.r] += obs[:self.r]
        multiplier = 2 * obs[self.r:self.r+self.i] - 1
        total_obs[self.r:] = obs[-self.i:] / 100 * multiplier
        total_reward += reward
        if done: break
      return total_obs, total_reward, done, info
  return Repeater

class LocalizeWrapper(gym.RewardWrapper):
  def _reward(self, a):
    d = np.diag(a) * (FLAGS.local_weight - 1)
    return np.mean(d + a, axis=1)

class SquishReward(gym.RewardWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.reward_size = 1
  def _reward(self, a):
    return np.mean(a)

def make_env():
  env = gym.make('traffic-v0')
  env.set_graph(GridRoad(1,1,250))
  env.seed_generator()
  env.reset_entrypoints()
  if FLAGS.render: env.rendering = True
  env = Repeater(FLAGS.light_iterations)(env)
  if FLAGS.warmup_lights > 0: env = WarmupWrapper(FLAGS.warmup_lights)(env)
  # if FLAGS.local_weight > 1: env = LocalizeWrapper(env)
  # if FLAGS.squish_rewards: env = SquishReward(env)
  return env

transients = ['mode', 'render', 'restore', 'save_settings']

def main(*_):
  if FLAGS.save_settings:
    with open('settings', 'w') as f:
      json.dump(FLAGS.__flags, f, indent=4, separators=(',', ': '))
  FLAGS.episode_len = int(FLAGS.episode_secs / FLAGS.light_secs)
  FLAGS.light_iterations = int(FLAGS.light_secs / FLAGS.rate)
  FLAGS.episode_ticks = int(FLAGS.episode_secs / FLAGS.rate)
  if FLAGS.mode == 'weights': FLAGS.restore = True
  if FLAGS.render: FLAGS.mode = 'validate'
  globals()[FLAGS.trainer].run(make_env)

if __name__ == '__main__':
  try:
    with open('settings', 'r') as f:
      for k, v in json.load(f).items():
        if k not in transients: setattr(FLAGS, k, v)
  except FileNotFoundError: pass
  tf.app.run()
