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
import tensorflow as tf
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd
import numpy as np
from util import *

# we should also try returning to a change/nochange scheme

# We should not printed discounted (for compatability sake)

# Okay, let's add an rnn to this. 

# Actually, it will be easier just to add some history.
# Let's shrink the obs_rate down to 2
# And store FLAGS.history extra observations


# why is he not learning?
# let's simplify the network

add_argument('--episode_secs', 600, type=int)
add_argument('--light_secs', 5, type=int)
add_argument('--warmup_lights', 5, type=int)
add_argument('--obs_rate', 2, type=int)
add_argument('--hist_size', 6, type=int)
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
    self.obs_size = FLAGS.obs_rate * 4
    self.hist_obs_size = FLAGS.hist_size * 4
    # self.shape = [self.i, ((FLAGS.obs_rate + FLAGS.hist_size) * 4 + 1)]
    self.shape = [self.i, 1]
    self.observation_space = Box(0, 1, shape=self.shape)
    # self.history = np.zeros((self.i, self.hist_obs_size))
  def _reset(self):
    super(Repeater, self)._reset()
    self.counter = 0
    # self.history[:] = 0
    self.hc = 0
    return np.zeros(self.shape)
  def _step(self, action):
    self.counter += 1
    done = False
    total_reward = 0
    detected = np.zeros((FLAGS.obs_rate, self.r), dtype=np.float32)
    elapsed_phase = np.zeros(self.i, dtype=np.float32)
    if FLAGS.mode != 'train':
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
    total_obs = elapsed_phase[:,None]
    # total_obs = np.concatenate((for_conv, self.history, elapsed_phase[:,None]), 1)
    # self.history[:, self.hc : self.hc + self.obs_size] = for_conv
    # self.hc += self.obs_size
    # if self.hc + self.obs_size > self.hist_obs_size: self.hc = 0
    total_reward += 1 / (np.sum(np.square(self.env.cars_on_roads())) + 1)
    done |= self.counter == FLAGS.episode_len
    # assert self.counter < FLAGS.episode_len + 1
    return total_obs, total_reward, done, info


# Something is going wrong we should learn the constant function. But on 3x3 we don't.

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
      
      obzr = tf.reshape(obz, [-1, np.prod(ob_space.shape)])
      last_out = obzr
      for i in range(3):
        last_out = tf.nn.tanh(U.dense(last_out, 5, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
      self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
      last_out = obzr
      for i in range(2):
        last_out = tf.nn.tanh(U.dense(last_out, 5, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
      pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

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

def model(name, ob_space, ac_space):
  return MyModel(name, ob_space, ac_space)

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

# What about showing the prob dists per light?
# Or showing the values of the weights and biases?

def run():
  args.parse_flags()
  args.apply_derivations(args.PARSER) 
  env = gym.make('traffic-v0')
  env.set_graph(GridRoad(3,3,250))
  env.seed_generator()
  env.reset_entrypoints()
  if FLAGS.render: env.rendering = True
  env = Repeater(env)
  if FLAGS.mode == 'train':
    sess = U.make_session(num_cpu=4)
    sess.__enter__()
    pposgd.learn(env, model, callback=saver,
        timesteps_per_batch=512, clip_param=0.2,
        max_timesteps=FLAGS.episode_len * 800,
        entcoeff=1e-6, optim_epochs=30, optim_stepsize=1e-3,
        optim_batchsize=256, gamma=0.99, lam=0.95, schedule='linear')
    U.save_state(SAVE_LOC)
    ws, bs = sess.run(["pi/polfinal/w:0", "pi/polfinal/b:0"])
    print("Weights", ws)
    print("Biases", bs)
  elif FLAGS.mode == 'const0':
    ones = np.ones(env.action_space.n)
    def episode():
      # env.unwrapped.reset_entrypoints()
      env.reset()
      for i in range(FLAGS.episode_len):
        o,r,d,info = env.step(ones)
        yield i,o,ones,r,info
        if d: break
    analyze(env, episode)
  elif FLAGS.mode == 'const1':
    zeros = np.zeros(env.action_space.n)
    def episode():
      # env.unwrapped.reset_entrypoints()
      env.reset()
      for i in range(FLAGS.episode_len):
        o,r,d,info = env.step(zeros)
        yield i,o,zeros,r,info
        if d: break
    analyze(env, episode)
  elif FLAGS.mode == 'fixed':
    def phase(i):
      return int((i % (FLAGS.spacing * 2)) >= FLAGS.spacing)
    actions = np.zeros((2, env.action_space.n))
    actions[1,:] = 1
    def episode():
      env.unwrapped.reset_entrypoints()
      env.reset()
      for i in range(FLAGS.episode_len):
        a = actions[phase(i)]
        o,r,d,info = env.step(a)
        if FLAGS.render: print("Obs", o)
        yield i,o,a,r,info
        if d: break
    analyze(env, episode)
  elif FLAGS.mode == 'validate':
    act = model("pi", env.observation_space, env.action_space).act
    sess = U.make_session(num_cpu=4)
    sess.__enter__()
    state = U.load_state(SAVE_LOC)
    ws, bs = sess.run(["pi/polfinal/w:0", "pi/polfinal/b:0"])
    print("Weights", ws)
    print("Biases", bs)
    def episode():
      env.unwrapped.reset_entrypoints()
      obs = env.reset()
      for t in range(FLAGS.episode_len):
        a = act(False, obs)[0]
        if FLAGS.render: print("Action:", a)
        new_obs, reward, done,info = env.step(a)
        if FLAGS.render: print("Obs", new_obs)
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
