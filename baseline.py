import gym
from gym.spaces import Box
import gym_traffic
from gym_traffic.envs.roadgraph import GridRoad
from gym_traffic.envs.traffic_env import RATE
import baselines.ppo1.pposgd_simple as pposgd
from baselines.ppo1.mlp_policy import MlpPolicy
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np
from util import *
import argparse

# We should allow comparison to constant (for one dir)
# Maybe it just knows the right thing to compare for each entry

# WE should also play with other grid topologies.
# One way streets would allow green ways to have an effect.
# but this structure is substantially harder to weight

# the weights really shouldn't be negative

# Make render rate faster. Nicer to look at non-jerky cars

# should perhaps jump to acktr

# lookup how timesteps_per_batch and optim_batchsize work

# Let's add Remi reward. Again:
  # -1 if no cars passed in the last 2 seconds, and people on the opposite phase
  # were waiting. 
  # +1 if cars passed in the last 2 seconds, and no people on the opposite phase were
  # waiting.
  
# using cem would also be nice here. Or ES.

# look @ understanding momentum post again for fiddling with learning rate

# add asymmetric flow functionality (arbitrary probs)


WARMUP_LIGHTS = 10
OBS_RATE = 2
LIGHT_SECS = 2
EPISODE_LIGHTS = 100
SPACING = 6
SAVE_LOC = "baselined/saved"

LIGHT_TICKS = int(LIGHT_SECS // RATE)
EPISODE_TICKS = int(EPISODE_LIGHTS * LIGHT_TICKS)
OBS_MOD = int(LIGHT_TICKS // OBS_RATE)

C = (SPACING * LIGHT_TICKS / 50)
F = 0.5

def episode(env, f):
  o = env.reset()
  for i in range(EPISODE_LIGHTS):
    a = f(i, o, env)
    o,r,d,info = env.step(a)
    yield i,o,a,r,info
    if d: break

def dist_layer(x, intersections):
  result = np.zeros((intersections * 4, intersections * 4), dtype=np.float32)
  m = int(np.sqrt(intersections))
  for i in range(1, intersections):
    if (i % m) > 0: # left
      prev = (i-1)*4
      result[prev,i] = 1
      result[prev+1,i] = -1
      result[prev+2,i] = -F
      result[prev+3,i] = F
    if (i % m) < (m-1): # right
      nxt = (i+1)*4
      result[nxt,intersections + i] = 1
      result[nxt+1,intersections + i] = -1
      result[nxt+2,intersections + i] = -F
      result[nxt+3,intersections + i] = F
    if i >= m: # bottom
      btm = (i - m)*4
      result[btm,i] = 1
      result[btm+1,i] = -1
      result[btm+2,i] = -F
      result[btm+3,i] = F
    if i < intersections - m: # top
      top = (i + m)*4
      result[top,i] = 1
      result[top+1,i] = -1
      result[top+2,i] = -F
      result[top+3,i] = F
  mask = (result != 0).astype(np.float32)
  weights = tf.get_variable("distw", initializer=result)
  return tf.nn.tanh(tf.matmul(x, weights * tf.constant(mask)))

def phase_layer(x, intersections):
  result = np.zeros((intersections * 4, intersections), dtype=np.float32)
  for i in range(intersections):
    cur = i*4
    result[cur,i] = -1
    result[cur+1,i] = 1
    result[cur+2,i] = C
    result[cur+3,i] = -C
  mask = (result != 0).astype(np.float32)
  weights = tf.get_variable("phasew", initializer=result)
  return tf.nn.tanh(tf.matmul(x, weights * tf.constant(mask)))

def comb_layer(x, intersections):
  result = np.zeros((intersections * 5, intersections), dtype=np.float32)
  m = int(np.sqrt(intersections))
  for i in range(intersections):
    if (i % m) == 0:
      result[4 * intersections + i, i] = 3
    else:
      result[i,i] = 1
  weights = tf.get_variable("combw", initializer=result)
  return tf.matmul(x, weights)

def elapsed_phases(obs, i):
  phase = obs[-2*i:-i]
  not_phase = 1 - phase
  delay = obs[-i:] / 50
  return np.stack((delay * phase, delay * not_phase, phase, not_phase), -1)

class Repeater(gym.Wrapper):
  def __init__(self, env, entry):
    super(Repeater, self).__init__(env)
    self.entry = entry
    self.rendering = env.rendering
    self.r = self.unwrapped.graph.train_roads
    self.i = self.unwrapped.graph.intersections
    self.shape = [self.i, 4 + (4 * OBS_RATE)]
    self.observation_space = Box(0, 1, shape=self.shape)
    self.zeros = np.zeros(self.i)

  def _reset(self):
    self.env.reset_entrypoints(self.entry)
    obs = super(Repeater, self)._reset()
    # self.env.seed_generator(0)
    self.counter = 0
    rendering = self.env.rendering
    self.env.rendering = False
    i = 0
    detected = np.zeros((OBS_RATE, self.r), dtype=np.float32)
    while i < WARMUP_LIGHTS:
      i += 1
      for j in range(LIGHT_TICKS - 1):
        obs, _, done, _ = self.env.step(
            self.zeros if j > 0 else self.env.action_space.sample())
        if done: break
        detected[j // OBS_MOD] += obs[:self.r]
      if done:
        print("Overflowed in warmup, trying again")
        self.env.reset()
        i = 0
    self.env.rendering = rendering
    reshaped = detected.reshape(OBS_RATE, self.i, 4)
    for_conv = reshaped.transpose((1, 0, 2)).reshape(self.i, -1)
    return np.concatenate((for_conv, elapsed_phases(obs, self.i)), axis=-1)

  def _step(self, action):
    self.counter += 1
    done = False
    total_reward = 0
    detected = np.zeros((OBS_RATE, self.r), dtype=np.float32)
    if not self.env.training:
      if self.env.steps == 0: change_times = []
      else:
        change = np.logical_xor(self.env.current_phase, action).astype(np.int32)
        light_dist = (self.env.elapsed + 1) * change.astype(np.int32)
        light_dist_secs = light_dist.astype(np.float32) * RATE
        change_times = light_dist_secs[np.nonzero(light_dist_secs)]
      info = {'light_times': change_times}
    else: info = None
    for it in range(LIGHT_TICKS):
      obs, reward, done, _ = self.env.step(action)
      total_reward += reward
      detected[it // OBS_MOD] += obs[:self.r]
      if done: break
    reshaped = detected.reshape(OBS_RATE, self.i, 4)
    for_conv = reshaped.transpose((1, 0, 2)).reshape(self.i, -1)
    total_obs = np.concatenate((for_conv, elapsed_phases(obs, self.i)), axis=-1)
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
      obz = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, *ob_space.shape])
      phaseinfo = tf.reshape(obz[:,:,-4:], [-1, intersections * 4])
      everything = tf.reshape(obz, [-1, intersections * features])

      dist = dist_layer(phaseinfo, intersections)
      phase = phase_layer(phaseinfo, intersections)
      last_out = tf.nn.tanh(U.dense(everything, 8 * intersections,
        "pffc1", weight_init=U.normc_initializer(1.0)))
      last_out = U.dense(last_out, intersections,
        "pffc2", weight_init=U.normc_initializer(1.0))
      pdparam = comb_layer(tf.concat((dist, phase), 1), intersections) + last_out

      last_out = everything
      growth = 5 * intersections
      for i in range(2):
        new_out = tf.nn.tanh(U.dense(last_out, growth, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        last_out = tf.concat((new_out, last_out), 1)
      self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

      self.pd = pdtype.pdfromflat(pdparam)
      self.state_in = []
      self.state_out = []

      stochastic = tf.placeholder(dtype=tf.bool, shape=())
      ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
      self._act = U.function([stochastic, obz], [ac, self.vpred])

  def act(self, stochastic, ob):
      ac1, vpred1 =  self._act(stochastic, ob[None])
      return ac1[0], vpred1[0]
  def get_variables(self):
      return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
  def get_trainable_variables(self):
      return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
  def get_initial_state(self):
      return []

BEST = 0
def saver(lcls, glbs):
  global BEST
  iters = lcls['iters_so_far']
  if iters == 0 and CONTINUED: U.load_state(SAVE_LOC)
  if iters == 0: U.save_state(SAVE_LOC)
  if iters > 0 and iters % 50 == 0:
    m = np.mean(lcls['rewbuffer'])
    if m > BEST:
      BEST = m
      U.save_state(SAVE_LOC)

def fixed_phase(i):
  return int((i % (SPACING * 2)) >= SPACING)

def run(env, mode, entry):
  env = Repeater(env, entry)
  fixed_ac = np.zeros((2, env.action_space.n))
  fixed_ac[1,:] = 1
  if mode == 'train':
    sess = U.make_session(num_cpu=4)
    sess.__enter__()
    pposgd.learn(env, MyModel, callback=saver,
        timesteps_per_batch=256, clip_param=0.2,
        max_timesteps=EPISODE_LIGHTS * 1e6,
        entcoeff=1e-4, optim_epochs=4, optim_stepsize=5e-4,
        optim_batchsize=64, gamma=0.99, lam=0.96) # schedule='linear')
    U.save_state(SAVE_LOC)
  elif mode == 'random':
    dist_plot('random', analyze(env,
      lambda: episode(env, lambda i,o,e: e.action_space.sample())))
  elif mode == 'const':
    zeros = np.zeros(env.action_space.n)
    dist_plot('const', analyze(env,
      lambda: episode(env, lambda i,o,e: zeros)))
  elif mode == 'fixed':
    dist_plot('fixed', analyze(env,
      lambda: episode(env, lambda i,o,e: fixed_ac[fixed_phase(i)])))
  elif mode == 'validate':
    act = MyModel("pi", env.observation_space, env.action_space).act
    sess = U.make_session(num_cpu=4)
    sess.__enter__()
    state = U.load_state(SAVE_LOC)
    trained_stats = analyze(env,
      lambda: episode(env, lambda i,o,e: act(False, o)[0]))
    fixed_stats = analyze(env, 
      lambda: episode(env, lambda i,o,e: fixed_ac[fixed_phase(i)]))
    pval_plot(trained_stats, fixed_stats)

def analyze(env, g):
  return print_running_stats(alot(lambda: episode_reward(env, g())))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--entry", default='all')
  parser.add_argument("--render", type=bool)
  parser.add_argument("--continued", type=bool)
  parser.add_argument("--mode", default='train')
  args = parser.parse_args()
  env = gym.make('traffic-v0')
  env.set_graph(GridRoad(3,3,250))
  env.seed_generator()
  env.rendering = args.render
  env.training = args.mode == 'train'
  CONTINUED = args.continued
  run(env, args.mode, args.entry)
