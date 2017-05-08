import tensorflow as tf
import numpy as np
from util import *
from args import FLAGS

def run(env_f):
  env = env_f()
  zeros = np.zeros(env.action_space.shape)
  def episode():
    env.reset()
    for i in range(FLAGS.episode_len):
      o,r,d,info = env.step(zeros)
      yield i,o,zeros,r,info
      if d: break
  print_running_stats(forever(lambda: episode_reward(episode())))
