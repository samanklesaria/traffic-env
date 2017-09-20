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

# How can we make it stronger in its convictions?
# Multiply by a constant

# okay, that works well.
# Now let's add a bit of sophistication,
# - linear combo of phase and follow
# - follow gives 4 numbers (top, bottom, left, right)
# - 3x3
# - do it in tensorflow
# - let it learn?
# - add in loop detector stuff (with history? or lstm?)

# ---

# Make render rate faster. Nicer to look at non-jerky cars

# Question: why are manually specified weights failing?

# should perhaps jump to acktr

# lookup how timesteps_per_batch and optim_batchsize work

# Still not working, which is curious.
# Let's add Remi reward. Again:
  # -1 if no cars passed in the last 2 seconds, and people on the opposite phase
  # were waiting. 
  # +1 if cars passed in the last 2 seconds, and no people on the opposite phase were
  # waiting.
  
# using cem would also be nice here. Or ES.

# look @ understanding momentum post again for fiddling with learning rate

# add asymmetric flow functionality (arbitrary probs)

# should also examine why stochastic action sucks (never more than 7/3 split)

# Review the operation of the alg, ensure that we are going through enough episodes

# Add intelligent initialization


WARMUP_LIGHTS = 0 # 10
OBS_RATE = 2
LIGHT_SECS = 2
EPISODE_LIGHTS = 100
SPACING = 12
SAVE_LOC = "baselined/saved"

LIGHT_TICKS = int(LIGHT_SECS // RATE)
EPISODE_TICKS = int(EPISODE_LIGHTS * LIGHT_TICKS)
OBS_MOD = int(LIGHT_TICKS // OBS_RATE)

C = (SPACING * LIGHT_TICKS / 50)
# print("C=", C)
F = 0.2

# Let's start out with everybody on fixed cycles. Replicate it again

# we should see how this looks. Set lr = 0 and train.

# This feels non standard, and has an icky extra parameter
# what if we find the distance, then tanh it?

# Clearly this isn't going how we imagined.
# What we need to do is restore

# should tf debug too

def dist_layer(x, intersections):
  result = np.zeros((intersections * 4, intersections))
  m = int(np.sqrt(intersections))
  for i in range(1, intersections):
    if (i % m) > 0:
      prev = (i-1)*4
      result[prev,i] = 1
      result[prev+1,i] = -1
      result[prev+2,i] = -F
      result[prev+3,i] = F
  return tf.matmul(x, tf.constant(result, dtype=tf.float32))

def phase_layer(x, intersections):
  result = np.zeros((intersections * 4, intersections))
  for i in range(intersections):
    cur = i*4
    result[cur,i] = -1
    result[cur+1,i] = 1
    result[cur+2,i] = C
    result[cur+3,i] = -C
  return tf.matmul(x, tf.constant(result, dtype=tf.float32))

def comb_layer(x, intersections):
  result = np.zeros((intersections * 2, intersections))
  m = int(np.sqrt(intersections))
  for i in range(intersections):
    if (i % M) == 0:
      result[intersections + i, i] = 5
    else:
      result[i,i] = 5
  return tf.matmul(x, tf.constant(combiner, dtype=tf.float32))

def elapsed_phases(obs, i):
  phase = obs[-2*i:-i]
  not_phase = 1 - phase
  delay = obs[-i:] / 50
  result = np.stack((delay * phase, delay * not_phase, phase, not_phase), -1)
  # print(result)
  return result

class Repeater(gym.Wrapper):
  def __init__(self, env):
    super(Repeater, self).__init__(env)
    self.rendering = env.rendering
    self.r = self.unwrapped.graph.train_roads
    self.i = self.unwrapped.graph.intersections
    # self.shape = [self.i, 3]
    # self.shape = [self.i, 3 + (4 * OBS_RATE)]
    self.shape = [self.i, 4]
    self.observation_space = Box(0, 1, shape=self.shape)
    self.zeros = np.zeros(self.i)

  def _reset(self):
    obs = super(Repeater, self)._reset()
    self.env.seed_generator(0)
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
    # reshaped = detected.reshape(OBS_RATE, self.i, 4)
    # for_conv = reshaped.transpose((1, 0, 2)).reshape(self.i, -1)
    # return np.concatenate((for_conv, elapsed_phases(obs, self.i)), axis=-1)
    return elapsed_phases(obs, self.i)

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
      obz = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, *ob_space.shape])
      obzr = tf.reshape(obz, [-1, intersections * features])

      last_out = obzr
      dist = dist_layer(obzr, intersections)
      phase = phase_layer(obzr, intersections)
      pdparam = comb_layer(tf.concat((dist, phase), 1), intersections)

      last_out = obzr
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
    # if name == "pi":
    #   # we'll need to make a whole new tboard file
    #   tf.summary.histogram("pol_w", tf.get_default_graph().get_tensor_by_name("pi/polfinal/w:0"))
    #   tf.summary.histogram("pol_b", tf.get_default_graph().get_tensor_by_name("pi/polfinal/b:0"))

  def act(self, stochastic, ob):
      ac1, vpred1 =  self._act(stochastic, ob[None])
      return ac1[0], vpred1[0]
  def get_variables(self):
      return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
  def get_trainable_variables(self):
      return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
  def get_initial_state(self):
      return []

INTER = 4
M = 2
dist2x2 = np.zeros((INTER * 4, INTER))
for i in range(INTER):
  if (i % M) > 0:
    prev = (i-1)*4
    dist2x2[prev,i] = 1
    dist2x2[prev+1,i] = -1
    dist2x2[prev+2,i] = -F
    dist2x2[prev+3,i] = F

phase2x2 = np.zeros((INTER * 4, INTER))
for i in range(INTER):
  cur = i*4
  phase2x2[cur,i] = -1
  phase2x2[cur+1,i] = 1
  phase2x2[cur+2,i] = C
  phase2x2[cur+3,i] = -C

combiner = np.zeros((INTER * 2, INTER))
for i in range(INTER):
  if (i % M) == 0:
    combiner[INTER+i,i] = 5
  else:
    combiner[i,i] = 5

def fake_nn(o):
  dist = np.tanh(o @ dist2x2)
  phase = np.tanh(o @ phase2x2)
  full = np.concatenate((dist, phase))
  score = (full @ combiner)
  # print("Score", score)
  return score > 0
# THE WHOLE THING IS LINEAR! MIND BLOWN!
# WE could either 1) unnecessarily pass through atan
# Or 2) combine all the matrices together
# Try both!


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

def run(env, mode, interactive):
  env = Repeater(env)
  if mode == 'train':
    sess = U.make_session(num_cpu=4)
    sess.__enter__()
    pposgd.learn(env, MyModel, callback=saver,
        timesteps_per_batch=256, clip_param=0.2,
        max_timesteps=EPISODE_LIGHTS * 1e5,
        entcoeff=1e-4, optim_epochs=4, optim_stepsize=5e-4,
        optim_batchsize=64, gamma=0.99, lam=0.96) # schedule='linear')
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
        o,r,d,info = env.step(zeros)
        yield i,o,zeros,r,info
        if d: break
    analyze(env, episode, interactive)
  elif mode == 'fixed':
    def phase(i):
      return int((i % (SPACING * 2)) >= SPACING)
    actions = np.zeros((2, env.action_space.n))
    actions[1,:] = 1
    def episode():
      o = env.reset()
      for i in range(EPISODE_LIGHTS):
        a = actions[phase(i)]
        # forprint = np.reshape(o, (9,4))
        # print("Seeing obs action", forprint)
        # print("Using action", a)
        o,r,d,info = env.step(a)
        yield i,o,a,r,info
        if d: break
    analyze(env, episode, interactive)
  elif mode == 'numpy':
    def episode():
      o = env.reset()
      for i in range(EPISODE_LIGHTS):
        # print("Seeing obs action", o)
        a = fake_nn(o.reshape(-1))
        # print("Using action", a)
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
        # forprint = np.reshape(obs, (9,4))
        # print("Seeing obs action", forprint)
        new_obs, reward, done,info = env.step(a)
        # if env.rendering: print("Obs", new_obs)
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
  parser.add_argument("--continued", type=bool)
  parser.add_argument("--mode", default='train')
  args = parser.parse_args()
  env = gym.make('traffic-v0')
  env.set_graph(GridRoad(2,2,250))
  env.seed_generator()
  env.reset_entrypoints(args.entry)
  env.rendering = args.render
  env.training = args.mode == 'train'
  CONTINUED = args.continued
  run(env, args.mode, args.interactive)
