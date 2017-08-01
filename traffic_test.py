import gym
import gym_traffic
from gym_traffic.spaces.gspace import GSpace
from gym_traffic.envs.roadgraph import GridRoad
from gym_traffic.wrappers.warmup import WarmupWrapper
import numpy as np
from args import parse_flags, add_argument, add_derivation, FLAGS
from alg_flags import run_alg

add_argument('--episode_secs', 600, type=int)
add_argument('--light_secs', 5, type=int)
add_argument('--warmup_lights', 5, type=int)
add_argument('--obs_rate', 5, type=int)

def secs_derivations():
  FLAGS.episode_len = int(FLAGS.episode_secs / FLAGS.light_secs)
  FLAGS.light_iterations = int(FLAGS.light_secs / FLAGS.rate)
  FLAGS.episode_ticks = int(FLAGS.episode_secs / FLAGS.rate)
add_derivation(secs_derivations)

class Repeater(gym.Wrapper):
  def __init__(self, env):
    super(Repeater, self).__init__(env)
    self.r = self.unwrapped.graph.train_roads
    self.i = self.unwrapped.graph.intersections
    self.observation_space = GSpace([self.i, FLAGS.obs_rate * 4 + 1], np.float32(1))
  def _reset(self):
    super(Repeater, self)._reset()
    return np.zeros(self.observation_space.shape)
  def _step(self, action):
    done = False
    total_reward = 0
    detected = np.zeros((FLAGS.obs_rate, self.r), dtype=np.float32)
    elapsed_phase = np.zeros(self.i, dtype=np.float32)
    if FLAGS.mode == 'validate':
      if self.env.steps == 0: change_times = []
      else:
        change = np.logical_xor(self.env.current_phase, action).astype(np.int32) 
        light_dist = (self.env.elapsed + 1) * change.astype(np.int32)
        light_dist_secs = light_dist.astype(np.float32) * FLAGS.rate
        change_times = light_dist_secs[np.nonzero(light_dist_secs)]
      info = {'light_times': change_times}
    else: info = None
    obs_modulus = FLAGS.light_iterations // FLAGS.obs_rate
    for it in range(FLAGS.light_iterations):
      obs, reward, done, _ = self.env.step(action)
      total_reward += reward
      if done: break
      detected[it // obs_modulus] += obs[:self.r]
    multiplier = 2 * obs[-2*self.i:-self.i] - 1
    elapsed_phase = obs[-self.i:] / 100 * multiplier 
    reshaped = detected.reshape(FLAGS.obs_rate, self.i, 4)
    for_conv = reshaped.transpose((1, 0, 2)).reshape(self.i, -1)
    total_obs = np.concatenate((for_conv, elapsed_phase[:,None]), 1)
    return total_obs, total_reward, done, info

def make_env():
  env = gym.make('traffic-v0')
  env.set_graph(GridRoad(3,3,250))
  env.seed_generator()
  env.reset_entrypoints()
  if FLAGS.render: env.rendering = True
  env = Repeater(env)
  if FLAGS.warmup_lights > 0: env = WarmupWrapper(FLAGS.warmup_lights)(env)
  return env

if __name__ == '__main__':
  parse_flags()
  run_alg(make_env)
