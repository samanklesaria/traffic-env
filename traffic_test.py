import gym
import gym_traffic
from gym_traffic.spaces.gspace import GSpace
from gym_traffic.envs.roadgraph import GridRoad
from gym_traffic.wrappers.warmup import WarmupWrapper
import tensorflow as tf
import numpy as np
import gym_traffic.algorithms.a3c as a3c
import gym_traffic.algorithms.random as random
import gym_traffic.algorithms.const0 as const0
import gym_traffic.algorithms.const1 as const1
import gym_traffic.algorithms.greedy as greedy
import gym_traffic.algorithms.qrnn as qrnn
import gym_traffic.algorithms.qlearn as qlearn
import gym_traffic.algorithms.spacedgreedy as spacedgreedy
import gym_traffic.algorithms.fixed as fixed
from args import parse_flags, add_argument, add_derivation, FLAGS
from gym_traffic.wrappers.history import HistoryWrapper

add_argument('--episode_secs', 600, type=int)
add_argument('--light_secs', 5, type=int)
add_argument('--warmup_lights', 2, type=int)
add_argument('--local_weight', 1, type=int)
add_argument('--squish_rewards', False, type=bool)
add_argument('--remi', True, type=bool)

def secs_derivations():
  FLAGS.episode_len = int(FLAGS.episode_secs / FLAGS.light_secs)
  FLAGS.light_iterations = int(FLAGS.light_secs / FLAGS.rate)
  FLAGS.episode_ticks = int(FLAGS.episode_secs / FLAGS.rate)
  if FLAGS.mode == 'weights': FLAGS.restore = True
  if FLAGS.render: FLAGS.mode = 'validate'
  if FLAGS.remi: FLAGS.reward_printing = "avg"
add_derivation(secs_derivations)

def Repeater(repeat_count):
  class Repeater(gym.Wrapper):
    def __init__(self, env):
      super(Repeater, self).__init__(env)
      self.r = self.unwrapped.graph.train_roads
      self.i = self.unwrapped.graph.intersections
      self.observation_space = GSpace(np.ones(2 * self.r + self.i, dtype=np.float32))
    def _step(self, action):
      done = False
      total_reward = 0
      total_obs = np.zeros_like(self.observation_space.limit)
      for _ in range(repeat_count):
        obs, reward, done, info = self.env.step(action)
        total_obs[:self.r] += obs[:self.r]
        total_obs[self.r:2*self.r] = obs[self.r:2*self.r]
        multiplier = 2 * obs[-2*self.i:-self.i] - 1
        total_obs[-self.i:] = obs[-self.i:] / 100 * multiplier
        total_reward += reward
        if done: break
      return total_obs, total_reward, done, info
  return Repeater

class Remi(gym.Wrapper):
  def _step(self, action):
    obs, _, done, info = self.env.step(action)
    r = self.unwrapped.remi_reward() 
    self.unwrapped.passed_dst[:] = False
    return obs, r, done, info
    
class LocalizeWrapper(gym.RewardWrapper):
  def _reward(self, a):
    d = np.diag(a) * (FLAGS.local_weight - 1)
    return np.mean(d + a, axis=1) / FLAGS.local_weight

class SquishReward(gym.RewardWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.reward_size = 1
  def _reward(self, a):
    return np.mean(a)

def make_env():
  env = gym.make('traffic-v0')
  env.set_graph(GridRoad(3,3,250))
  env.seed_generator()
  env.reset_entrypoints()
  if FLAGS.render: env.rendering = True
  env = Repeater(FLAGS.light_iterations)(env)
  if FLAGS.warmup_lights > 0: env = WarmupWrapper(FLAGS.warmup_lights)(env)
  if FLAGS.remi: env = Remi(env)
  if FLAGS.local_weight > 1: env = LocalizeWrapper(env)
  if FLAGS.squish_rewards: env = SquishReward(env)
  if FLAGS.history > 1: env = HistoryWrapper(FLAGS.history)(env)
  return env

if __name__ == '__main__':
  parse_flags()
  globals()[FLAGS.trainer].run(make_env)
