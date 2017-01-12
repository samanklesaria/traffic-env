import gym
import gym_traffic
from gym_traffic.envs.traffic_env import GridRoad
from gym_traffic.wrappers.warmup import WarmupWrapper
from gym_traffic.wrappers.history import HistoryWrapper
from gym_traffic.wrappers.strobe import StrobeWrapper
import tensorflow as tf
import numpy as np
import json
import gym_traffic.algorithms.a3c as trainer
# import gym_traffic.algorithms.polgrad as trainer
# import gym_traffic.algorithms.cem as trainer
# import gym_traffic.algorithms.random as trainer

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('episode_secs', 180, 'Secs per episode')
flags.DEFINE_integer('light_secs', 4, 'Seconds per light')
flags.DEFINE_integer('warmup_lights', 0, 'Number of lights to choose randomly')
flags.DEFINE_integer('local_weight', 3, 'Weight to give local elements')

the_grid = GridRoad(3,3,1)
the_grid.generate_entrypoints(0)

class LocalizeWrapper(gym.RewardWrapper):
  def _reward(self, a):
    n = a.shape[0]
    fullcounts = np.tile(a, (n,1))
    weighted = (np.diag(a) * (FLAGS.local_weight - 1) + fullcounts) / FLAGS.local_weight
    return np.sum(weighted, axis=1)

def make_env():
  light_iterations = int(FLAGS.light_secs / FLAGS.rate)
  env = gym.make('traffic-v0')
  env.set_graph(the_grid)
  if FLAGS.render: env.rendering = True
  return HistoryWrapper(4)(StrobeWrapper(light_iterations, 4)(
    LocalizeWrapper(env)))

def main(_):
  FLAGS.episode_len = int(FLAGS.episode_secs / FLAGS.light_secs)
  FLAGS.cars_per_sec = FLAGS.global_cars_per_sec * the_grid.m
  with open('settings', 'w') as f:
    json.dump(FLAGS.__flags, f, indent=4, separators=(',', ': '))
  trainer.run(make_env)

if __name__ == '__main__':
  tf.app.run()
