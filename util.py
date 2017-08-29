from itertools import count
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def square(x): return x*x

def forever(f):
  while True: yield f()

def print_running_stats(iterator):
  trip_times = []
  light_times = []
  unfinished = []
  overflowed = 0
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
        unfinished.append(info['unfinished'])
        overflowed += int(info['overflowed'])
  except KeyboardInterrupt:
    print("Interrupted")
    return (light_times, trip_times, unfinished, overflowed / iterations)

def make_subplot(ax, data):
  ax.hist(data, color='c')
  ax.axvline(np.mean(data), color='b', linestyle='dashed',linewidth=2)

def make_plot(light_times, trip_times, unfinished):
  fig = plt.figure()
  fig.suptitle("Stats", fontweight='bold', fontsize=14)
  fig.subplots_adjust(hspace=0.5)
  ax = fig.add_subplot(311)
  ax.set_title("Light Times")
  make_subplot(ax, light_times)
  ax = fig.add_subplot(312)
  ax.set_title("Trip Times")
  make_subplot(ax, trip_times)
  ax = fig.add_subplot(313)
  ax.set_title("Unfinished")
  make_subplot(ax, unfinished)

def write_data(light_times, trip_times, unfinished, overflowed):
  make_plot(light_times, trip_times, unfinished)
  plt.savefig('hist.png')
  np.save("light_times.npy", light_times)
  np.save("trip_times.npy", trip_times)
  np.save("unfinished.npy", unfinished)
  print("Light times mean %2f, mode %2f, std %2f" % (np.mean(light_times), stats.mode(light_times, axis=None).mode, np.std(light_times)))
  print("Trip times mean %2f, mode %2f, std %2f" % (np.mean(trip_times), stats.mode(trip_times, axis=None).mode, np.std(trip_times)))
  print("Unfinished mean %2f, mode %2f, std %2f" % (np.mean(unfinished), stats.mode(unfinished, axis=None).mode, np.std(unfinished)))
  print("Overflowed:", overflowed)

def display_data(light_times, trip_times, unfinished, overflowed):
  make_plot(light_times, trip_times, unfinished)
  plt.show()
  print("Light times mean %2f, mode %2f, std %2f" % (np.mean(light_times), stats.mode(light_times, axis=None).mode, np.std(light_times)))
  print("Trip times mean %2f, mode %2f, std %2f" % (np.mean(trip_times), stats.mode(trip_times, axis=None).mode, np.std(trip_times)))
  print("Unfinished mean %2f, mode %2f, std %2f" % (np.mean(unfinished), stats.mode(unfinished, axis=None).mode, np.std(unfinished)))
  print("Overflowed:", overflowed)

def episode_reward(env, gen):
  num_0s = 0
  num_1s = 0
  reward = 0.0
  light_times = []
  for (i,_,a,r,info,*_) in gen:
    reward += r 
    light_times.extend(info['light_times'])
    nz = np.count_nonzero(a)
    num_1s += nz
    num_0s += (a.size - nz)
  total_actions = num_1s + num_0s
  info_struct = {'zerop': num_0s / total_actions, 'light_times': light_times,
    'onep': num_1s / total_actions, 'trip_times': env.unwrapped.triptimes(),
    'overflowed': env.unwrapped.overflowed,
    'unfinished': np.sum(env.unwrapped.cars_on_roads())}
  return (reward, info_struct)

