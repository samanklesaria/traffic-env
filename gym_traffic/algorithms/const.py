import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env_f):
  env = env_f()
  iterations = 0
  reward_sum = 0
  zeros = np.ones(env.action_space.shape)
  while True:
    multiplier = 1
    iterations += 1
    obs = env.reset()
    for _ in range(FLAGS.episode_len):
      obs, reward, done, _ = env.step(zeros)
      reward_sum += np.mean(reward)
      multiplier *= FLAGS.gamma
      if done: break
    print(reward_sum / iterations)
