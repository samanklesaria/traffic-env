import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env_f):
  FLAGS.obs_deltas = False
  env = env_f()
  iterations = 0
  reward_sum = 0
  dims = env.observation_space.shape[-2:]
  while True:
    multiplier = 1
    iterations += 1
    obs = env.reset()
    for i in range(FLAGS.episode_len):
      flat_obs = np.reshape(obs, [-1, 4, *dims])
      important = flat_obs[-1].transpose([1,2,0])
      action = (important.dot([1,1,-1,-1]) < 0).astype(np.int8).reshape(-1)
      obs, reward, done, _ = env.step(action)
      reward_sum += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
      multiplier *= FLAGS.gamma
      if done: break
    # print("Iterations:", i)
    # print("Cars generated:", env.unwrapped.generated_cars)
    print("Avg reward", reward_sum / iterations)
