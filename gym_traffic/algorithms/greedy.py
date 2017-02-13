import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

def run(env_f):
  FLAGS.obs_deltas = False
  FLAGS.learn_switch = False
  env = env_f()
  iterations = 0
  reward_mean = 0
  reward_var = 0
  dims = env.observation_space.shape[-2:]
  try:
    while True:
      multiplier = 1
      iterations += 1
      total_reward = 0
      obs = env.reset()
      for i in range(FLAGS.episode_len):
        flat_obs = np.reshape(obs, [-1, 4, *dims])
        important = flat_obs[-1].transpose([1,2,0])
        action = (important.dot([1,1,-1,-1]) < 0).astype(np.int8).reshape(-1)
        obs, reward, done, _ = env.step(action)
        total_reward += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
        multiplier *= FLAGS.gamma
        if done: break
      reward_mean = (total_reward + (iterations - 1) * reward_mean) / iterations
      if iterations >= 2:
        reward_var = (iterations - 2) / (iterations - 1) * reward_var + \
            np.square(total_reward - reward_mean) / iterations
      print("Total reward", total_reward)
      print("Cars generated:", env.unwrapped.generated_cars)
  except KeyboardInterrupt:
    print("Avg reward", reward_mean)
    print("Reward stddev", np.sqrt(reward_var))
    raise
