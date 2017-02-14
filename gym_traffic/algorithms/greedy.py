import tensorflow as tf
import numpy as np
from util import print_running_stats

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env_f):
  FLAGS.learn_switch = False
  FLAGS.process_obs = False
  env = env_f()
  dims = env.observation_space.shape[-2:]
  def rewards():
    while True:
      multiplier = 1
      total_reward = 0
      obs = env.reset()
      for i in range(FLAGS.episode_len):
        flat_obs = np.reshape(obs, [-1, 10, *dims])
        important = flat_obs[-1,4:8].transpose([1,2,0])
        action = env.action_space.to_action(important.dot([1,1,-1,-1]) < 0)
        obs, reward, done, _ = env.step(action)
        total_reward += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
        multiplier *= FLAGS.gamma
        if done: break
      yield total_reward
  print_running_stats(rewards())
