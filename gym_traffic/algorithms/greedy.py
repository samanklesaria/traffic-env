import tensorflow as tf
import numpy as np
from util import print_running_stats

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env_f):
  FLAGS.learn_switch = False
  # FLAGS.process_obs = False
  env = env_f()
  def rewards():
    while True:
      multiplier = 1
      total_reward = 0
      obs = env.reset()
      for i in range(FLAGS.episode_len):
        if FLAGS.render: print("OBS", obs)
        action = env.action_space.to_action(obs[:,:,:4].dot([1,1,-1,-1]) < 0)
        obs, reward, done, _ = env.step(action)
        if FLAGS.render: print("REWARD", reward)
        total_reward += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
        multiplier *= FLAGS.gamma
        if done: break
      yield total_reward
  print_running_stats(rewards())
