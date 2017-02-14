import tensorflow as tf
import numpy as np
from util import print_running_stats

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env_f):
  env = env_f()
  ones = np.ones(env.action_space.shape)
  def rewards():
    while True:
      reward_sum = 0
      multiplier = 1
      obs = env.reset()
      for _ in range(FLAGS.episode_len):
        obs, reward, done, _ = env.step(ones)
        reward_sum += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
        multiplier *= FLAGS.gamma
        if done: break
      yield reward_sum
  print_running_stats(rewards())
