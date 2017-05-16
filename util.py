from itertools import count
import math
import numpy as np
from args import FLAGS
import matplotlib.pyplot as plt

def square(x): return x*x

def forever(f):
  while True: yield f()

def print_running_stats(iterator):
  trip_times = []
  light_times = []
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
        light_times.extend(info['light_times'])
  except KeyboardInterrupt:
    print("Interrupted")
    return (light_times, trip_times)

def make_subplot(ax, data):
  ax.hist(data, color='c')
  ax.axvline(np.mean(data), color='b', linestyle='dashed',linewidth=2)

def make_plot(light_times, trip_times):
  fig = plt.figure()
  fig.suptitle("Stats for " + FLAGS.trainer, fontweight='bold', fontsize=14)
  fig.subplots_adjust(hspace=0.5)
  ax = fig.add_subplot(211)
  ax.set_title("Light Times")
  make_subplot(ax, light_times)
  ax = fig.add_subplot(212)
  ax.set_title("Trip Times")
  make_subplot(ax, trip_times)

def write_data(light_times, trip_times):
  make_plot(light_times, trip_times)
  plt.savefig('hist.png')
  np.save("light_times.npy", light_times)
  np.save("trip_times.npy", trip_times)

def episode_reward(gen):
  num_0s = 0
  num_1s = 0
  reward = 0.0
  multiplier = 1.0
  light_times = []
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
    info_struct = {'zerop': num_0s / total_actions, 'light_times': light_times,
      'onep': num_1s / total_actions, 'trip_times': info['trip_times']}
  else: info_struct = None
  return (reward / denom, info_struct)

