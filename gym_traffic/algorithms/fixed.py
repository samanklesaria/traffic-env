import tensorflow as tf
import numpy as np
from util import *
from args import FLAGS

def phase(i):
  return int((i % (FLAGS.spacing * 2)) >= FLAGS.spacing)

def run(env_f):
  FLAGS.learn_switch = False
  env = env_f()
  actions = np.zeros((2, *env.action_space.shape))
  actions[1,:] = 1
  def episode():
    obs = env.reset()
    for i in range(FLAGS.episode_len):
      a = actions[phase(i)]
      obs, reward, done, info = env.step(a)
      yield i,obs,a,reward,info
      if done: break
  print_running_stats(forever(lambda: episode_reward(episode())))
