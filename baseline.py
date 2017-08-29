import gym
from gym.spaces import Box
import gym_traffic
from gym_traffic.envs.roadgraph import GridRoad
from gym_traffic.envs.traffic_env import RATE
import baselines.pposgd.pposgd_simple as pposgd
from baselines.pposgd.mlp_policy import MlpPolicy
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np
from util import *
import argparse

# We're dominating 1x1 one way.
# In symmetric flow, we're slightly worse (1x1). Not statistically significant difference.

# look @ understanding momentum post again for fiddling with learning rate

# add asymmetric flow functionality (arbitrary probs)

# Try another layer with weight sharing
# Get it to work for constant flow on 3x3
# Get it to learn fixed switching for symmetric flow on 1x1
# Get it to learn fixed switching for symmetric flow on 3x3
# Add loop detectors. Ensure performance doesn't drop
# Shoehorn in an rnn if you can

# should also examine why stochastic action sucks (never more than 7/3 split)

# Adapt to use tensorboard

# we could do weight sharing again. 
# the last layer is currently taking all nodes into account. we could do it separately

# Eventually we need to have shape [intersections, 3 + (4 * obs_rate)]

# Review the operation of the alg, ensure that we are going through enough episodes

WARMUP_LIGHTS = 10
OBS_RATE = 2
LIGHT_SECS = 5
EPISODE_LIGHTS = 100
SPACING = 3
SAVE_LOC = "baselined/saved"

LIGHT_TICKS = int(LIGHT_SECS // RATE)
EPISODE_TICKS = int(EPISODE_LIGHTS * LIGHT_TICKS)
OBS_MOD = int(LIGHT_TICKS // OBS_RATE)

def elapsed_phases(obs, i):
  phase = obs[-2*i:-i]
  not_phase = 1 - phase
  return np.stack((obs[-i:] / 50, phase, not_phase), -1)

class Repeater(gym.Wrapper):
  def __init__(self, env):
    super(Repeater, self).__init__(env)
    self.r = self.unwrapped.graph.train_roads
    self.i = self.unwrapped.graph.intersections
    self.shape = [self.i, 3]
    self.observation_space = Box(0, 1, shape=self.shape)
    self.zeros = np.zeros(self.i)

  def _reset(self):
    super(Repeater, self)._reset()
    self.counter = 0
    rendering = self.env.rendering
    self.env.rendering = False
    i = 0
    while i < WARMUP_LIGHTS:
      i += 1
      obs, _, done, _ = self.env.step(self.env.action_space.sample())
      for j in range(LIGHT_TICKS - 1):
        if done: break
        obs, _, done, _ = self.env.step(self.zeros)
      if done:
        print("Overflowed in warmup, trying again")
        self.env.reset()
        i = 0
    self.env.rendering = rendering
    return elapsed_phases(obs, self.i)

  def _step(self, action):
    self.counter += 1
    done = False
    total_reward = 0
    detected = np.zeros((OBS_RATE, self.r), dtype=np.float32)
    if not self.env.training:
      if self.env.steps == 0: change_times = []
      else:
        light_dist = (self.env.elapsed + 1) * action.astype(np.int32)
        light_dist_secs = light_dist.astype(np.float32) * RATE
        change_times = light_dist_secs[np.nonzero(light_dist_secs)]
      info = {'light_times': change_times}
    else: info = None
    for it in range(LIGHT_TICKS - 1):
      obs, reward, done, _ = self.env.step(self.zeros if it > 0 else action)
      total_reward += reward
      detected[(it+1) // OBS_MOD] += obs[:self.r]
      if done: break
    # reshaped = detected.reshape(OBS_RATE, self.i, 4)
    # for_conv = reshaped.transpose((1, 0, 2)).reshape(self.i, -1)
    # total_obs = np.concatenate((for_conv, elapsed_phases(obs, self.i)), axis=-1)
    total_obs = elapsed_phases(obs, self.i)
    total_reward += 1 / (np.sum(np.square(self.env.cars_on_roads())) + 1)
    done |= self.counter == EPISODE_LIGHTS
    return total_obs, total_reward, done, info

class MyModel:
  recurrent = False
  def __init__(self, name, ob_space, ac_space):
    intersections, features = ob_space.shape
    with tf.variable_scope(name):
      self.scope = tf.get_variable_scope().name
      self.pdtype = pdtype = make_pdtype(ac_space)
      ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, *ob_space.shape])
      with tf.variable_scope("obfilter"):
          self.ob_rms = RunningMeanStd(shape=ob_space.shape)
      # obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
      obz = ob
      
      # obzr = tf.reshape(obz, [-1, features])
      obzr = tf.reshape(obz, [-1, np.prod(ob_space.shape)])
      last_out = obzr
      for i in range(1):
        last_out = tf.nn.tanh(U.dense(last_out, 8, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
      # rout = tf.reshape(last_out, [-1, 8 * intersections])
      rout = last_out
      self.vpred = U.dense(rout, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
      last_out = obzr
      for i in range(1):
        last_out = tf.nn.tanh(U.dense(last_out, 8, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
      # rout = tf.reshape(last_out, [-1, 8 * intersections])
      rout = last_out
      pdparam = U.dense(rout, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

      # robz = tf.reshape(obz, [-1, features])
      #
      # last_out = robz
      # for i in range(3):
      #   last_out = tf.nn.tanh(U.dense(last_out, 30, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
      # last_out = tf.reshape(last_out, [-1, intersections * 30])
      # self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
      #
      # last_out = robz
      # for i in range(3):
      #   last_out = tf.nn.tanh(U.dense(last_out, 30, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
      # last_out = tf.reshape(last_out, [-1, intersections * 30])
      # pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

      self.pd = pdtype.pdfromflat(pdparam)
      self.state_in = []
      self.state_out = []

      stochastic = tf.placeholder(dtype=tf.bool, shape=())
      ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
      self._act = U.function([stochastic, ob], [ac, self.vpred])

  def act(self, stochastic, ob):
      ac1, vpred1 =  self._act(stochastic, ob[None])
      return ac1[0], vpred1[0]
  def get_variables(self):
      return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
  def get_trainable_variables(self):
      return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
  def get_initial_state(self):
      return []

def saver(lcls, glbs):
  iters = lcls['iters_so_far']
  if iters > 0 and iters % 100 == 0:
    U.save_state(SAVE_LOC)

def run(env, mode, interactive):
  env = Repeater(env)
  if mode == 'train':
    sess = U.make_session(num_cpu=4)
    sess.__enter__()
    pposgd.learn(env, MyModel, callback=saver,
        timesteps_per_batch=512, clip_param=0.2,
        max_timesteps=EPISODE_LIGHTS * 1000,
        entcoeff=1e-3, optim_epochs=30, optim_stepsize=1e-3,
        optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear')
    U.save_state(SAVE_LOC)
  elif mode == 'random':
    def episode():
      env.reset()
      for i in range(EPISODE_LIGHTS):
        a = env.action_space.sample()
        o,r,d,info = env.step(a)
        yield i,o,a,r,info
        if d: break
    analyze(env, episode, interactive)
  elif mode == 'const':
    zeros = np.zeros(env.action_space.n)
    def episode():
      env.reset()
      for i in range(EPISODE_LIGHTS):
        action = env.unwrapped.current_phase
        o,r,d,info = env.step(action)
        yield i,o,zeros,r,info
        if d: break
    analyze(env, episode, interactive)
  elif mode == 'fixed':
    def phase(i):
      return int((i % (SPACING * 2)) >= SPACING)
    actions = np.zeros((2, env.action_space.n))
    actions[1,:] = 1
    def episode():
      env.reset()
      for i in range(EPISODE_LIGHTS):
        a = np.logical_xor(env.unwrapped.current_phase, actions[phase(i)]).astype(np.int32)
        o,r,d,info = env.step(a)
        yield i,o,a,r,info
        if d: break
    analyze(env, episode, interactive)
  elif mode == 'validate':
    act = MyModel("pi", env.observation_space, env.action_space).act
    sess = U.make_session(num_cpu=4)
    sess.__enter__()
    state = U.load_state(SAVE_LOC)
    def episode():
      obs = env.reset()
      for t in range(EPISODE_LIGHTS):
        a = act(False, obs)[0]
        if env.rendering: print("Action:", a)
        new_obs, reward, done,info = env.step(a)
        if env.rendering: print("Obs", new_obs)
        yield t,obs,a,reward,info,new_obs,done
        if done: break
        obs = new_obs
    analyze(env, episode, interactive)

def analyze(env, g, interactive):
  data = print_running_stats(forever(lambda: episode_reward(env, g())))
  if interactive: return data
  write_data(*data)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--entry", default='all')
  parser.add_argument("--render", type=bool)
  parser.add_argument("--interactive", type=bool)
  parser.add_argument("--mode", default='train')
  args = parser.parse_args()
  env = gym.make('traffic-v0')
  env.set_graph(GridRoad(1,1,250))
  env.seed_generator()
  env.reset_entrypoints(args.entry)
  env.rendering = args.render
  env.training = args.mode == 'train'
  run(env, args.mode, args.interactive)
