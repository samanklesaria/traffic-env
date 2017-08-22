import gym
from gym.spaces import Box
import gym_traffic
from gym_traffic.envs.roadgraph import GridRoad
import baselines.pposgd.pposgd_simple as pposgd
from baselines.pposgd.mlp_policy import MlpPolicy
from args import FLAGS, add_argument
import args
import alg_flags
import baselines.common.tf_util as U
import numpy as np
from util import *

# The only error we get is underflows
# but shouldn't that generate 0, not nan?
# where are the nans coming from?
np.seterr(divide='raise', over='raise', invalid='raise')

# It's just choosing 1 all the time. Something is wrong. 

add_argument('--episode_secs', 600, type=int)
add_argument('--light_secs', 5, type=int)
add_argument('--warmup_lights', 5, type=int)
add_argument('--obs_rate', 5, type=int)
SAVE_LOC = "baselined/saved"

def secs_derivations():
  FLAGS.episode_len = int(FLAGS.episode_secs / FLAGS.light_secs)
  FLAGS.light_iterations = int(FLAGS.light_secs / FLAGS.rate)
  FLAGS.episode_ticks = int(FLAGS.episode_secs / FLAGS.rate)
args.add_derivation(secs_derivations)

class Repeater(gym.Wrapper):
  def __init__(self, env):
    super(Repeater, self).__init__(env)
    self.r = self.unwrapped.graph.train_roads
    self.i = self.unwrapped.graph.intersections
    self.shape = self.i * (FLAGS.obs_rate * 4 + 1)
    self.observation_space = Box(0, 1, shape=self.shape)
  def _reset(self):
    super(Repeater, self)._reset()
    self.counter = 0
    return np.zeros(self.shape)
  def _step(self, action):
    self.counter += 1
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
    total_reward += 1 / (np.sum(np.square(self.env.cars_on_roads())) + 1)
    done |= self.counter == FLAGS.episode_len
    assert self.counter < FLAGS.episode_len + 1
    return total_obs.reshape(-1), total_reward, done, info

def model(name, ob_space, ac_space):
  return MlpPolicy(name, ob_space, ac_space, 30, 3)


# let's try 1x1 again?
# this model doesn't share anything. We've made better.

def saver(lcls, glbs):
  iters = lcls['iters_so_far']
  if iters > 0 and iters % 100 == 0:
    U.save_state(SAVE_LOC)

# Adapt to use tensorboard
# Shall we adapt it to use a cnn?
# why did everything become nan?
# What does the bench.Monitor stuff do?

# What exactly is max_timesteps doing?
# Ensure that you fully understand the alg

def run():
  args.parse_flags()
  args.apply_derivations(args.PARSER) 
  env = gym.make('traffic-v0')
  env.set_graph(GridRoad(3,3,250))
  env.seed_generator()
  env.reset_entrypoints()
  env = Repeater(env)
  if FLAGS.mode == 'train':
    sess = U.make_session(num_cpu=4)
    sess.__enter__()
    pposgd.learn(env, model, callback=saver,
        timesteps_per_batch=256, clip_param=0.2,
        max_timesteps=FLAGS.episode_len * 10000,
        entcoeff=0.01, optim_epochs=30, optim_stepsize=1e-3,
        optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear')
    U.save_state(SAVE_LOC)
  elif FLAGS.mode == 'const0':
    ones = np.ones(env.action_space.shape)
    def episode():
      env.unwrapped.reset_entrypoints()
      env.reset()
      for i in range(FLAGS.episode_len):
        o,r,d,info = env.step(ones)
        yield i,o,ones,r,info
        if d: break
    analyze(env, episode)
  elif FLAGS.mode == 'const1':
    zeros = np.zeros(env.action_space.shape)
    def episode():
      env.unwrapped.reset_entrypoints()
      env.reset()
      for i in range(FLAGS.episode_len):
        o,r,d,info = env.step(zeros)
        yield i,o,ones,r,info
        if d: break
    analyze(env, episode)
  elif FLAGS.mode == 'fixed':
    def phase(i):
      return int((i % (FLAGS.spacing * 2)) >= FLAGS.spacing)
    actions = np.zeros((2, *env.action_space.shape))
    actions[1,:] = 1
    def episode():
      env.unwrapped.reset_entrypoints()
      env.reset()
      for i in range(FLAGS.episode_len):
        a = actions[phase(i)]
        o,r,d,info = env.step(a)
        yield i,o,a,r,info
        if d: break
    analyze(env, episode)
  elif FLAGS.mode == 'validate':
    act = model("pi", env.observation_space, env.action_space).act
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    state = U.load_state(SAVE_LOC)
    def episode():
      env.unwrapped.reset_entrypoints()
      obs = env.reset()
      for t in range(FLAGS.episode_len):
        a = act(True, obs)[0]
        new_obs, reward, done,info = env.step(a)
        yield t,obs,a,reward,info,new_obs,done
        if done: break
        obs = new_obs
    analyze(env, episode)

def analyze(env, g):
  data = print_running_stats(forever(lambda: episode_reward(env, g())))
  if FLAGS.interactive: return data
  write_data(*data)

if __name__ == '__main__':
  run()
