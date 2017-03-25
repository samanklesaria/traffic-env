import tensorflow as tf
import numpy as np
from util import print_running_stats

flags = tf.app.flags
FLAGS = flags.FLAGS

def phase(i):
  return int((i % 6) >= 3)

def run(env_f):
  FLAGS.learn_switch = False
  env = env_f()
  actions = np.zeros((2, *env.action_space.shape))
  actions[1,:] = 1
  def rewards():
    while True:
      multiplier = 1
      reward_sum = 0
      obs = env.reset()
      for i in range(FLAGS.episode_len):
        if FLAGS.render: print("OBS", obs)
        if FLAGS.render: print("Action", actions[phase(i)])
        obs, reward, done, _ = env.step(actions[phase(i)])
        if FLAGS.render: print("REW", reward)
        reward_sum += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
        multiplier *= FLAGS.gamma
        if done: break
      yield reward_sum
  print_running_stats(rewards())
