from itertools import count
import math

def square(x): return x*x

def print_running_stats(iterator):
  reward_mean = 0
  reward_var = 0
  for iterations in count(1):
    reward = next(iterator)
    reward_mean = (reward + (iterations - 1) * reward_mean) / iterations
    if iterations >= 2:
      reward_var = (iterations - 2) / (iterations - 1) * reward_var + \
          square(reward - reward_mean) / iterations
    print("Reward %2f\t Mean %2f\t Std %2f" % (reward, reward_mean, math.sqrt(reward_var)))
