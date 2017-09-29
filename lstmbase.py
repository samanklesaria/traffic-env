import gym
import argparse
import numpy as np
from gym.spaces import Box
from baselines.a2c.a2c import learn
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.a2c.utils import fc, batch_to_seq, seq_to_batch, lstm, sample, check_shape

import gym_traffic
from gym_traffic.envs.roadgraph import GridRoad
from gym_traffic.envs.traffic_env import RATE
from util import *
import tensorflow as tf


class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            rX = tf.reshape(X, [-1, nh*nw*nc])
            h = fc(tf.cast(rX, tf.float32)/255., 'f1', 32, act=tf.nn.tanh, init_scale=np.sqrt(2))
            h2 = fc(h, 'f2', 64, act=tf.nn.tanh, init_scale=np.sqrt(2))
            h3 = fc(h2, 'f3', 512, act=tf.nn.tanh, init_scale=np.sqrt(2))
            xs = batch_to_seq(h3, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value




# Right. 1 problem is that the lstm arch uses convs that
# are way too big. 
# Let's just make a dumb port that doesn't use cnns

WARMUP_LIGHTS = 10
OBS_RATE = 2
LIGHT_SECS = 2
EPISODE_LIGHTS = 100
SPACING = 6

LIGHT_TICKS = int(LIGHT_SECS // RATE)
EPISODE_TICKS = int(EPISODE_LIGHTS * LIGHT_TICKS)
OBS_MOD = int(LIGHT_TICKS // OBS_RATE)

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
    g = self.unwrapped.graph
    self.r = g.train_roads
    self.i = g.intersections
    self.shape = [g.m, g.n, 4 + (4 * OBS_RATE)]
    self.observation_space = Box(0, 1, shape=self.shape)
    self.zeros = np.zeros(self.i)

  def _reset(self):
    self.env.reset_entrypoints(self.entry)
    obs = super(Repeater, self)._reset()
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
    result = np.concatenate((for_conv, elapsed_phases(obs, self.i)), axis=-1)
    return np.reshape(result, self.shape)

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
    total_flat_obs = np.concatenate((for_conv, elapsed_phases(obs, self.i)), axis=-1)
    total_obs = np.reshape(total_flat_obs, self.shape)
    total_reward += 1 / (np.sum(np.square(self.env.cars_on_roads())) + 1)
    done |= self.counter == EPISODE_LIGHTS
    return total_obs, total_reward, done, info

def run(make_env, mode, seed):
  set_global_seeds(seed)
  if mode == 'train':
    env = SubprocVecEnv([make_env(seed + i) for i in range(4)])
    learn(LstmPolicy, env, seed, total_timesteps=EPISODE_LIGHTS * int(1e5),
        lrschedule='constant')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--entry", default='all')
  parser.add_argument("--render", type=bool)
  parser.add_argument("--continued", type=bool)
  parser.add_argument("--mode", default='train')
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  def make_env(rank):
    def _thunk():
      env = gym.make('traffic-v0')
      env.set_graph(GridRoad(3,3,250))
      env.seed(rank)
      env.rendering = args.render
      env.training = args.mode == 'train'
      return Repeater(env, args.entry)
    return _thunk
  run(make_env, args.mode, args.seed)
