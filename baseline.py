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

# Another curious thing: in restore, we get a reward of 0.9, while in training
# with params that don't change, we get reward of 0.4. Is it the stochastic switch?
# YES. It's definitely the stochastic switch. 

# Okay, now we're doing the same thing, but the weights are trainable. 

# Still not working, which is curious.
# Let's add Remi reward. Again:
  # -1 if no cars passed in the last 2 seconds, and people on the opposite phase
  # were waiting. 
  # +1 if cars passed in the last 2 seconds, and no people on the opposite phase were
  # waiting.
  
  # Do this after a little bit.

# using cem would also be nice here. Or ES.

# Should at least compare against feature engineered version, 
# using gradient descent to find the right params for phase length and offset.
# It will be a linear model, so easy to understand, visualize, etc.

# To make a feature engineered version, just don't use U.dense. Make your own.

# If we wanted to do feature engineering:
# change = w*elapsed - d1*phase1 - d2*phase2

# to make this constant phase 2, we just have 
# w=0, d1 = 1, d2 = 0

# What if we wanted to offset nearby lights?
# We'd want to change n after our neighbor changed.
# (elapsed_neighbor - n)
# ok, no conv
# The weights would be the same, as the rates are the same
# We could make the light interval smaller (2 secs)
# The light in top left would have change = w*elapsed - d1*phase1 - d2*phase2
# The lights adjacent to it would have change = sum a*(elapsed_neighbor_k - d_k)
# The only node that benefits from conv is middle in 3x3. So fc is probably best for now

# We're dominating 1x1 one way.
# In symmetric flow, we're slightly worse (1x1). Not statistically significant difference.
# It feels like we get worse when weight sharing EVEN IN 1x1 CASE which is odd.

# look @ understanding momentum post again for fiddling with learning rate

# add asymmetric flow functionality (arbitrary probs)

# Try another layer with weight sharing
# Get it to work for constant flow on 3x3
# Get it to learn fixed switching for symmetric flow on 1x1
# Get it to learn fixed switching for symmetric flow on 3x3
# Add loop detectors. Ensure performance doesn't drop
# Shoehorn in an rnn if you can

# should also examine why stochastic action sucks (never more than 7/3 split)

# Eventually we need to have shape [intersections, 3 + (4 * obs_rate)]

# Review the operation of the alg, ensure that we are going through enough episodes

WARMUP_LIGHTS = 10
OBS_RATE = 2
LIGHT_SECS = 2
EPISODE_LIGHTS = 100
SPACING = 12
SAVE_LOC = "baselined/saved"

LIGHT_TICKS = int(LIGHT_SECS // RATE)
EPISODE_TICKS = int(EPISODE_LIGHTS * LIGHT_TICKS)
OBS_MOD = int(LIGHT_TICKS // OBS_RATE)

C = -(SPACING * LIGHT_TICKS / 50)

def cinit(shape, dtype=None, partition_info=None):
  return tf.constant([C,C], dtype=tf.float32)

# Okay. Let's try to get this to learn C.
# Great. now the two phases are independent. 
# Now let's try to do phase offsets.
# Work out how the offsets should be ideally to find best initialization.

def elapsed_phases(obs, i):
  phase = obs[-2*i:-i]
  not_phase = 1 - phase
  delay = obs[-i:] / 50
  return np.stack((delay * phase, delay * not_phase, phase, not_phase), -1)

class Repeater(gym.Wrapper):
  def __init__(self, env):
    super(Repeater, self).__init__(env)
    self.r = self.unwrapped.graph.train_roads
    self.i = self.unwrapped.graph.intersections
    # self.shape = [self.i, 3]
    # self.shape = [self.i, 3 + (4 * OBS_RATE)]
    self.shape = [self.i, 4]
    self.observation_space = Box(0, 1, shape=self.shape)
    self.zeros = np.zeros(self.i)

  def _reset(self):
    super(Repeater, self)._reset()
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
        light_dist = (self.env.elapsed + 1) * action.astype(np.int32)
        light_dist_secs = light_dist.astype(np.float32) * RATE
        change_times = light_dist_secs[np.nonzero(light_dist_secs)]
      info = {'light_times': change_times}
    else: info = None
    for it in range(LIGHT_TICKS):
      obs, reward, done, _ = self.env.step(self.zeros if it > 0 else action)
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

      # Wait- we've been stupid. We can't multiply our inputs together. We can only
      # add them. 
      # we could do a 2x2 conv with a stride. Or go back to the old conv
      last_out = tf.reshape(obz, [-1, intersections * features])
      growth = 10
      # for i in range(2):
      #   new_out = tf.nn.tanh(U.dense(last_out, growth, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
      #   last_out = tf.concat((new_out, last_out), 1)
      #   features += growth
      self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]


      # Okay, the reward was -1.55. WHY?
      # Ideally we'd step through in real time. 
      # That's a little more challenging here. 

      # For the moment, let's just print it
      # optimal = tf.constant([1, 1, C, C])
      last_out = tf.reshape(obz, [-1, 4])
      # for i in range(2):
      #   new_out = tf.nn.tanh(U.dense(last_out, growth, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
      #   last_out = tf.concat((new_out, last_out), 1)
      # dummy = U.dense(last_out, 1, "polfinal", U.normc_initializer(0.01))

      w = tf.get_variable("polfc/w", [2], initializer=cinit)
      optimal = tf.concat((tf.constant([1,1], dtype=tf.float32), w), 0)
      dummy = tf.reduce_sum(optimal * last_out, axis=1)
      pdparam = tf.reshape(dummy, [-1, intersections])
      # pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

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

def saver(lcls, glbs):
  iters = lcls['iters_so_far']
  if iters > 0 and iters % 100 == 0:
    U.save_state(SAVE_LOC)

# lookup how timesteps_per_batch and optim_batchsize work

def run(env, mode, interactive):
  env = Repeater(env)
  if mode == 'train':
    sess = U.make_session(num_cpu=4)
    sess.__enter__()
    pposgd.learn(env, MyModel, callback=saver,
        timesteps_per_batch=256, clip_param=0.2,
        max_timesteps=EPISODE_LIGHTS * 1e4,
        entcoeff=1e-5, optim_epochs=4, optim_stepsize=1e-3,
        optim_batchsize=64, gamma=0.99, lam=0.96, schedule='linear')
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
      o = env.reset()
      for i in range(EPISODE_LIGHTS):
        a = np.logical_xor(env.unwrapped.current_phase, actions[phase(i)]).astype(np.int32)
        # forprint = np.reshape(o, (9,4))
        # print("Seeing obs action", forprint)
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
    optimal = np.array([1, 1, C, C])
    def episode():
      obs = env.reset()
      for t in range(EPISODE_LIGHTS):
        a = act(False, obs)[0]
        if env.rendering: print("Action:", a)
        # forprint = np.reshape(obs, (9,4))
        # print("Seeing obs action", forprint)
        # print("Choosing action", a)
        # print("Should be", forprint.dot(optimal))
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
  env.set_graph(GridRoad(3,3,250))
  env.seed_generator()
  env.reset_entrypoints(args.entry)
  env.rendering = args.render
  env.training = args.mode == 'train'
  run(env, args.mode, args.interactive)
