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
  trip_times = []
  try:
    reward_mean = 0
    reward_var = 0
    for iterations in count(1):
      reward, info = next(iterator)
      reward_mean = (reward + (iterations - 1) * reward_mean) / iterations
      if iterations >= 2:
        reward_var = (iterations - 2) / (iterations - 1) * reward_var + \
            square(reward - reward_mean) / iterations
      print("Reward %2f\t Mean %2f\t Std %2f" % (reward, reward_mean, math.sqrt(reward_var)))
      if info:
        print("One prob: %2f,\t Zero prob: %2f" % (info['onep'], info['zerop']))
        trip_times.extend(info['trip_times'])
  finally:
    if FLAGS.mode == 'validate':
      print("Writing histograms")
      plt.hist(light_times, color='c')
      light_time_mean = np.mean(light_times)
      plt.axvline(light_time_mean, color='b', linestyle='dashed',linewidth=2)
      plt.figtext(0.02, 0.02, "Mean %2f, std %2f" % (light_time_mean, np.std(light_times)))
      plt.savefig('light_times.png')
      plt.clf()
      plt.hist(trip_times, color='c')
      trip_time_mean = np.mean(trip_times)
      plt.axvline(trip_time_mean, color='b', linestyle='dashed', linewidth=2)
      plt.figtext(0.02, 0.02, "Mean %2f, std %2f" % (trip_time_mean, np.std(trip_times)))
      plt.savefig('trip_times.png')
      np.save("light_times.npy", light_times)
      np.save("trip_times.npy", trip_times)

def episode_reward(gen):
  num_0s = 0
  num_1s = 0
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
  if FLAGS.mode == 'validate':
    total_actions = num_1s + num_0s
    info_struct = {'zerop': num_0s / total_actions,
      'onep': num_1s / total_actions, 'trip_times': info['trip_times']}
  else: info_struct = None
  return (reward / denom, info_struct)

