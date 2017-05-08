from itertools import count
import math
import numpy as np
from args import FLAGS
import matplotlib.pyplot as plt

def square(x): return x*x

def forever(f):
  while True: yield f()

light_times = []

def print_running_stats(iterator):
  try:
    reward_mean = 0
    reward_var = 0
    for iterations in count(1):
      reward = next(iterator)
      reward_mean = (reward + (iterations - 1) * reward_mean) / iterations
      if iterations >= 2:
        reward_var = (iterations - 2) / (iterations - 1) * reward_var + \
            square(reward - reward_mean) / iterations
      print("Reward %2f\t Mean %2f\t Std %2f" % (reward, reward_mean, math.sqrt(reward_var)))
  finally:
    if light_times and FLAGS.mode == 'validate':
      print("Writing histograms")
      plt.hist(light_times)
      plt.savefig('light_times.png')

def episode_reward(gen):
  reward = 0.0
  multiplier = 1.0
  for (i,_,_,r,info,*_) in gen:
    reward += np.mean(r) * (multiplier if FLAGS.print_discounted else 1)
    multiplier *= FLAGS.gamma
    if info: light_times.extend(info['light_times'])
  if not FLAGS.print_avg:
    denom = 1
  elif FLAGS.gamma == 1:
    denom = i+1
  else:
    denom = (math.pow(FLAGS.gamma, i+1) - 1) / (FLAGS.gamma - 1) 
  return reward / denom

