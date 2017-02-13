import tensorflow as tf
import numpy as np
from itertools import count

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env_f):
  env = env_f()
  iterations = 0
  reward_mean = 0
  reward_var = 0
  try:
    for iterations in count(1):
      multiplier = 1
      iterations += 1
      total_reward = 0
      obs = env.reset()
      for _ in range(FLAGS.episode_len):
        obs, reward, done, _ = env.step(env.action_space.sample())
        total_reward += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
        multiplier *= FLAGS.gamma
        if done: break
      print(total_reward)
      reward_mean = (total_reward + (iterations - 1) * reward_mean) / iterations
      if iterations >= 2:
        reward_var = (iterations - 2) / (iterations - 1) * reward_var + \
            np.square(total_reward - reward_mean) / iterations
  except KeyboardInterrupt:
    print("Avg reward", reward_mean)
    print("Reward stddev", np.sqrt(reward_var))
    raise
