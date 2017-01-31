from gym_traffic.algorithms import in_dir
import traffic_test
import tensorflow as tf
import itertools
import numpy as np
import random

flags = tf.app.flags
FLAGS = flags.FLAGS

def main(_):
  for i in itertools.count(0):
    with in_dir("test_" + str(i)):
      print("Starting test", i)
      tf.reset_default_graph()
      FLAGS.save_settings = True
      FLAGS.light_secs = np.random.randint(1, 7)
      FLAGS.warmup_lights = np.random.randint(0, 8)
      FLAGS.cooldown = np.random.uniform(0, 0.5)
      FLAGS.waiting_penalty = 0 if bool(random.getrandbits(1)) else np.random.uniform(0, 0.5)
      FLAGS.local_weight = np.random.randint(1, 4)
      FLAGS.learning_rate = 10**np.random.uniform(-5, -2)
      FLAGS.gamma = np.random.uniform(0.95, 0.99)
      FLAGS.exploration = random.choice(['proportional', 'e_greedy'])
      FLAGS.poisson = bool(random.getrandbits(1))
      FLAGS.normalize_obs = bool(random.getrandbits(1))
      FLAGS.change_penalty = 50 * np.random.randint(0, 2)
      FLAGS.change_threshold = 45 // FLAGS.light_secs
      traffic_test.main()

if __name__ == '__main__':
  tf.app.run()

