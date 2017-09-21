from itertools import count
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def square(x): return x*x

def alot(f):
  for _ in range(300): yield f()

def print_running_stats(iterator):
  trip_times = []
  light_times = []
  unfinished = []
  rewards = []
  overflowed = 0
  try:
    reward_mean = 0
    reward_var = 0
    for iterations in count(1):
      reward, info = next(iterator)
      rewards.append(reward)
      reward_mean = (reward + (iterations - 1) * reward_mean) / iterations
      if iterations >= 2:
        reward_var = (iterations - 2) / (iterations - 1) * reward_var + \
            square(reward - reward_mean) / iterations
      print("Reward %2f\t Mean %2f\t Std %2f" % (reward, reward_mean, math.sqrt(reward_var)))
      if info:
        print("One prob: %2f,\t Zero prob: %2f" % (info['onep'], info['zerop']))
        print("Change prob: %2f,\t Nochange prob: %2f" % (info['changep'], info['nochangep']))
        trip_times.extend(info['trip_times'])
        light_times.extend(info['light_times'])
        unfinished.append(info['unfinished'])
        overflowed += int(info['overflowed'])
  except (KeyboardInterrupt, StopIteration):
    overflowed /= iterations
    print("Interrupted\n")
    print("Light times mean %2f, mode %2f, std %2f" % (np.mean(light_times), stats.mode(light_times, axis=None).mode, np.std(light_times)))
    print("Trip times mean %2f, mode %2f, std %2f" % (np.mean(trip_times), stats.mode(trip_times, axis=None).mode, np.std(trip_times)))
    print("Unfinished mean %2f, mode %2f, std %2f" % (np.mean(unfinished), stats.mode(unfinished, axis=None).mode, np.std(unfinished)))
    print("Overflowed:", overflowed)
    return (light_times, trip_times, unfinished, rewards, overflowed)

colors = ['c', 'y']
lc = ['g', 'r']
def make_subplot(ax, datas, n):
  for i, data in enumerate(datas):
    x = data[n]
    ax.hist(x, color=colors[i])
    ax.axvline(np.mean(x), color=lc[i], linestyle='dashed',linewidth=2)

def make_plot(*infos):
  fig = plt.figure()
  fig.suptitle("Stats", fontweight='bold', fontsize=14)
  fig.subplots_adjust(hspace=0.9)
  ax = fig.add_subplot(411)
  ax.set_title("Light Times")
  make_subplot(ax, infos, 0)
  ax = fig.add_subplot(412)
  ax.set_title("Trip Times")
  make_subplot(ax, infos, 1)
  ax = fig.add_subplot(413)
  ax.set_title("Unfinished")
  make_subplot(ax, infos, 2)
  ax = fig.add_subplot(414)
  ax.set_title("Rewards")
  make_subplot(ax, infos, 3)

def dist_plot(fname, info):
  make_plot(info)
  plt.savefig(fname + '.png')

def pval_plot(trained, fixed):
  make_plot(trained, fixed)
  plt.savefig("comparison.png")
  print("Rewared p-val", stats.ttest_ind(trained[3], fixed[3], equal_var=False)[1])
  print("Trip-times p-val", stats.ttest_ind(trained[1], fixed[1], equal_var=False)[1])

def episode_reward(env, gen):
  num_nochange = 0
  num_change = 0
  num_1s = 0
  num_0s = 0
  reward = 0.0
  light_times = []
  for (i,_,a,r,info,*_) in gen:
    reward += r 
    light_times.extend(info['light_times'])
    nz = np.count_nonzero(env.unwrapped.current_phase)
    num_1s += nz
    num_0s += (a.size - nz)
    nz = np.count_nonzero(a)
    num_change += nz
    num_nochange += (a.size - nz)
  total_actions = num_1s + num_0s
  info_struct = {'zerop': num_0s / total_actions, 'light_times': light_times,
    'onep': num_1s / total_actions, 'changep': num_change / total_actions,
    'nochangep': num_nochange / total_actions,
    'trip_times': env.unwrapped.triptimes(),
    'overflowed': env.unwrapped.overflowed,
    'unfinished': np.sum(env.unwrapped.cars_on_roads())}
  return (reward, info_struct)

