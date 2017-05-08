import tensorflow as tf
import numpy as np
from util import print_running_stats
from args import FLAGS

def run(env_f):
  env = env_f()
  ones = np.ones(env.action_space.shape)
  def episode():
    env.reset()
    for i in range(FLAGS.episode_len):
      o,r,d,info = env.step(ones)
      yield i,o,ones,r,info
      if d: break
  print_running_stats(forever(lambda: episode_reward(episode())))
