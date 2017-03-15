import tensorflow as tf
import numpy as np
from util import print_running_stats

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env_f):
  FLAGS.learn_switch = False
  env = env_f()
  def rewards():
    while True:
      multiplier = 1
      total_reward = 0
      obs = env.reset()
      for _ in range(FLAGS.episode_len):
        if FLAGS.render: print("OBS", obs)
        obs, reward, done, _ = env.step(env.action_space.sample())
        # if FLAGS.render: print("REWARD", reward)
        total_reward += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
        multiplier *= FLAGS.gamma
        if done: break
      yield total_reward
  print_running_stats(rewards())
