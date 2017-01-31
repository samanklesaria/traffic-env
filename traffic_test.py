import gym
import gym_traffic
from gym_traffic.envs.roadgraph import GridRoad, by_intersection
from gym_traffic.wrappers.warmup import WarmupWrapper
from gym_traffic.wrappers.history import HistoryWrapper
from gym_traffic.wrappers.strobe import StrobeWrapper
import tensorflow as tf
import numpy as np
import json
import gym_traffic.algorithms.a3c_rnn as a3c_rnn
import gym_traffic.algorithms.a3c as a3c
import gym_traffic.algorithms.a3c_conv as a3c_conv
import gym_traffic.algorithms.polgrad as polgrad
import gym_traffic.algorithms.polgrad_conv as polgrad_conv
import gym_traffic.algorithms.cem as cem
import gym_traffic.algorithms.random as random
import gym_traffic.algorithms.const as const
import gym_traffic.algorithms.greedy as greedy

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('episode_secs', 1000, 'Secs per episode')
flags.DEFINE_integer('light_secs', 5, 'Seconds per light')
flags.DEFINE_float('warmup_lights', 1, 'Number of lights to choose randomly')
flags.DEFINE_float('cooldown', 0, 'Portion of simulation time without introducing new agents')
flags.DEFINE_integer('local_weight', 0, 'Weight to give local elements')
flags.DEFINE_float('change_penality', 0, "How much should the network be penalized for no phase switch?")
flags.DEFINE_integer('change_threshold', 15, "After how many iterations should penality for no change apply?")
flags.DEFINE_integer('history', 4, "How many lights worth of history do we include?")
flags.DEFINE_float('reward_counts', 1, "How much should the network be rewarded for passing cars?")
flags.DEFINE_float('waiting_penalty', 0, "How much should the network be penalized for waiting cars?")
flags.DEFINE_boolean('normalize_obs', False, "Should we normalize observations?")
flags.DEFINE_boolean('scale_rewards', True, "Should we scale rewards?")

EPS = 1e-8

class LocalizeWrapper(gym.RewardWrapper):
  def _reward(self, a):
    d = np.diag(a) * (FLAGS.local_weight - 1)
    return np.sum(d + (1 - 1 / FLAGS.local_weight) * np.ones_like(d), axis=1)

class ScaleWrapper(gym.RewardWrapper):
  def _reward(self, reward):
    return reward / 50

class WaitingWrapper(gym.Wrapper):
  def _step(self, action):
    obs, reward, done, info = self.env.step(action)
    return obs, reward - FLAGS.waiting_penalty * \
        by_intersection(self.unwrapped.graph.intersections,
            self.unwrapped.graph.dests, np.reshape(obs, -1)), done, info

class ChangeWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.prev_action = None
    self.action_counts = np.zeros(env.action_space.shape, dtype=np.int)
  def _step(self, actionlist):
    action = np.array(actionlist)
    obs, reward, done, info = self.env.step(action)
    if self.prev_action is not None:
      same = (action == self.prev_action).astype(np.int)
      self.action_counts = self.action_counts * same + same
      over_threshold = (self.action_counts > FLAGS.change_threshold).astype(np.float32)
    else: over_threshold = 0
    self.prev_action = action
    return obs, FLAGS.reward_counts * reward - FLAGS.change_penality * over_threshold, done, info

class ObsNormWrapper(gym.ObservationWrapper):
  def _observation(self, o):
    o -= np.mean(o)
    o /= (np.std(o) + EPS)
    return o

def make_env():
  env = gym.make('traffic-v0')
  env.set_graph(GridRoad(4,4,8))
  if FLAGS.render: env.rendering = True
  if FLAGS.waiting_penalty > 0: env = WaitingWrapper(env)
  env = StrobeWrapper(FLAGS.light_iterations, 5)(env)
  if FLAGS.warmup_lights > 0: env = WarmupWrapper(FLAGS.warmup_lights)(env)
  if FLAGS.normalize_obs: env = ObsNormWrapper(env)
  if FLAGS.change_penality > 0: env = ChangeWrapper(env)
  if FLAGS.local_weight > 1: env = LocalizeWrapper(env)
  if FLAGS.scale_rewards: env = ScaleWrapper(env)
  if FLAGS.history > 0: env = HistoryWrapper(3)(env)
  return env

transients = ['validate', 'render', 'restore', 'weights', 'save_settings']

def main(*_):
  if FLAGS.save_settings:
    with open('settings', 'w') as f:
      json.dump(FLAGS.__flags, f, indent=4, separators=(',', ': '))
  FLAGS.episode_len = int(FLAGS.episode_secs / FLAGS.light_secs)
  FLAGS.cars_per_sec = FLAGS.local_cars_per_sec * 12 # THIS IS A PROBLEM!
  FLAGS.light_iterations = int(FLAGS.light_secs / FLAGS.rate)
  FLAGS.episode_ticks = int(FLAGS.episode_secs / FLAGS.rate)
  if FLAGS.weights: FLAGS.restore = True
  globals()[FLAGS.trainer].run(make_env)

if __name__ == '__main__':
  try:
    with open('settings', 'r') as f:
      for k, v in json.load(f).items():
        if k not in transients: setattr(FLAGS, k, v)
  except FileNotFoundError: pass
  tf.app.run()
