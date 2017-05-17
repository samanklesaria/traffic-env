import tensorflow as tf
import numpy as np
from util import *
from args import FLAGS

def run(env_f):
  FLAGS.learn_switch = False
  env = env_f()
  def episode():
    obs = env.reset()
    for i in range(FLAGS.episode_len):
      a = env.action_space.sample()
      obs, reward, done, info = env.step(a)
      if FLAGS.render: print("REWARD", reward)
      yield i,obs,a,reward,info
      if done: break
  data = print_running_stats(forever(lambda: episode_reward(env, episode())))
  if FLAGS.interactive: return data
  write_data(*data)
