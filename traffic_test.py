import gym
import gym_traffic
from gym_traffic.spaces.gspace import GSpace
from gym_traffic.envs.roadgraph import GridRoad
from gym_traffic.wrappers.warmup import WarmupWrapper
import numpy as np
from args import parse_flags, add_argument, add_derivation, FLAGS
from alg_flags import run_alg
from gym_traffic.wrappers.history import HistoryWrapper

add_argument('--episode_secs', 600, type=int)
add_argument('--light_secs', 5, type=int)
add_argument('--warmup_lights', 0, type=int)
add_argument('--local_weight', 1, type=int)
add_argument('--squish_rewards', False, type=bool)
add_argument('--remi', True, type=bool)

def secs_derivations():
  FLAGS.episode_len = int(FLAGS.episode_secs / FLAGS.light_secs)
  FLAGS.light_iterations = int(FLAGS.light_secs / FLAGS.rate)
  FLAGS.episode_ticks = int(FLAGS.episode_secs / FLAGS.rate)
  if FLAGS.trainer == 'polgrad_rnn': FLAGS.use_avg = True
add_derivation(secs_derivations)

# eventually this should be incorperated into traffic-env
def Repeater(repeat_count):
  class Repeater(gym.Wrapper):
    def __init__(self, env):
      super(Repeater, self).__init__(env)
      self.r = self.unwrapped.graph.train_roads
      self.i = self.unwrapped.graph.intersections
      self.observation_space = GSpace([2*self.r+self.i], np.float32(1))
    def _reset(self):
      super(Repeater, self)._reset()
      return self._step(self.action_space.sample())[0]
    def _step(self, action):
      done = False
      total_reward = 0
      total_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
      if FLAGS.mode == 'validate':
        change = np.logical_xor(self.env.current_phase, action).astype(np.int32) 
        light_dist = (self.env.elapsed + 1) * change.astype(np.int32)
        light_dist_secs = light_dist.astype(np.float32) / 2
        change_times = light_dist_secs[np.nonzero(light_dist_secs)]
        info = {'light_times': change_times}
      else: info = None
      for _ in range(repeat_count):
        obs, reward, done, _ = self.env.step(action)
        total_obs[:self.r] += obs[:self.r] # passed in the last interval
        total_obs[self.r:2*self.r] = obs[self.r:2*self.r] # detected currently
        multiplier = 2 * obs[-2*self.i:-self.i] - 1
        total_obs[-self.i:] = obs[-self.i:] / 100 * multiplier # elapsed time and phase
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
  run_alg(make_env)
