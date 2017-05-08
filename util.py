from itertools import count
import math
import numpy as np
from args import FLAGS
import matplotlib.pyplot as plt

def square(x): return x*x

def forever(f):
  while True: yield f()

# Global variables for global stats
light_times = []
num_1s = 0
num_0s = 0

def print_running_stats(iterator):
  global num_1s
  global num_0s
  try:
    reward_mean = 0
    reward_var = 0
    for iterations in count(1):
      num_0s = 0
      num_1s = 0
      reward = next(iterator)
      reward_mean = (reward + (iterations - 1) * reward_mean) / iterations
      if iterations >= 2:
        reward_var = (iterations - 2) / (iterations - 1) * reward_var + \
            square(reward - reward_mean) / iterations
      print("Reward %2f\t Mean %2f\t Std %2f" % (reward, reward_mean, math.sqrt(reward_var)))
      total_actions = num_1s + num_0s
      print("One prob: %2f,\t Zero prob: %2f" % (num_1s / total_actions, num_0s / total_actions))
  finally:
    if light_times and FLAGS.mode == 'validate':
      print("Writing histograms")
      plt.hist(light_times)
      plt.savefig('light_times.png')

def episode_reward(gen):
  global num_0s
  global num_1s
  reward = 0.0
  multiplier = 1.0
  for (i,_,_,r,info,*_) in gen:
    reward += np.mean(r) * (multiplier if FLAGS.print_discounted else 1)
    multiplier *= FLAGS.gamma
    if info:
      light_times.extend(info['light_times'])
      nz = np.count_nonzero(info['action'])
      num_1s += nz
      num_0s += (len(info['action']) - nz)
  if not FLAGS.print_avg:
    denom = 1
  elif FLAGS.gamma == 1:
    denom = i+1
  else:
    denom = (math.pow(FLAGS.gamma, i+1) - 1) / (FLAGS.gamma - 1) 
  return reward / denom

