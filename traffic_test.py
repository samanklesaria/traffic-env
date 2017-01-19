import gym
import gym_traffic
from gym_traffic.envs.roadgraph import GridRoad, roads_to_intersections
from gym_traffic.wrappers.warmup import WarmupWrapper
from gym_traffic.wrappers.history import HistoryWrapper
from gym_traffic.wrappers.strobe import StrobeWrapper
import tensorflow as tf
import numpy as np
import json
import gym_traffic.algorithms.a3c_rnn as a3c
import gym_traffic.algorithms.polgrad as polgrad
import gym_traffic.algorithms.cem as cem
import gym_traffic.algorithms.random as random

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('episode_secs', 300, 'Secs per episode')
flags.DEFINE_integer('light_secs', 2, 'Seconds per light')
flags.DEFINE_float('warmup_lights', 0, 'Number of lights to choose randomly')
flags.DEFINE_integer('cooldown', 0, 'Portion of simulation time without introducing new agents')
flags.DEFINE_integer('local_weight', 2, 'Weight to give local elements')
flags.DEFINE_float('change_penality', 0, "How much should the network be penalized for no phase switch?")
flags.DEFINE_integer('change_threshold', 15, "After how many iterations should penality for no change apply?")
flags.DEFINE_float('reward_counts', 1, "How much should the network be rewarded for passing cars?")
flags.DEFINE_float('waiting_penalty', 0, "How much should the network be penalized for waiting cars?")
flags.DEFINE_boolean('normalize_obs', True, "Should we normalize observations?")

class LocalizeWrapper(gym.RewardWrapper):
  def _reward(self, a):
    d = np.diag(a) * (FLAGS.local_weight - 1)
    return np.sum(d + (1 - 1 / FLAGS.local_weight) * np.ones_like(d), axis=1)

class ScaleWrapper(gym.RewardWrapper):
  def _reward(self, a):
    return a / 300.0

class ChangeWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.prev_action = None
    self.action_counts = np.zeros(env.action_space.shape, dtype=np.int)
  def _step(self, action):
    obs, reward, done, info = self.env.step(action)
    if self.prev_action is not None:
      same = (action == self.prev_action).astype(np.int)
      self.action_counts = self.action_counts * same + same
      over_threshold = (self.action_counts > FLAGS.change_threshold).astype(np.float32)
    else: over_threshold = 0
    self.prev_action = action
    return obs, FLAGS.reward_counts * reward - FLAGS.change_penality * over_threshold, done, info

EPS = 1e-8
class ObsNormWrapper(gym.ObservationWrapper):
  def _observation(self, o):
    o -= np.mean(o)
    o /= (np.std(o) + EPS)
    return o

def make_env(norender=False, randomized=True):
  the_grid = GridRoad(3,3,2)
  env = gym.make('traffic-v0')
  env.set_graph(the_grid, FLAGS.episode_ticks,
    cooldown=int(FLAGS.cooldown * FLAGS.episode_ticks), randomized=randomized)
  if FLAGS.render and not norender: env.rendering = True
  env = StrobeWrapper(FLAGS.light_iterations, 2)(env)
  if FLAGS.normalize_obs: env = ObsNormWrapper(env)
  if FLAGS.change_penality: env = ChangeWrapper(env)
  if FLAGS.local_weight > 1: env = LocalizeWrapper(env)
  return ScaleWrapper(env)

def main(*_):
  FLAGS.episode_len = int(FLAGS.episode_secs / FLAGS.light_secs)
  FLAGS.cars_per_sec = FLAGS.local_cars_per_sec * 3
  FLAGS.light_iterations = int(FLAGS.light_secs / FLAGS.rate)
  FLAGS.episode_ticks = int(FLAGS.episode_secs / FLAGS.rate)
  with open('settings', 'w') as f:
    json.dump(FLAGS.__flags, f, indent=4, separators=(',', ': '))
  globals()[FLAGS.trainer].run(make_env)

if __name__ == '__main__':
  tf.app.run()
