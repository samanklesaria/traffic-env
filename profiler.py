import tensorflow as tf
import cProfile
import traffic_test

flags = tf.app.flags
FLAGS = flags.FLAGS

def profile_it():
  global env
  for _ in range(50):
    env.reset()
    for _ in range(FLAGS.episode_len):
      env.step(env.action_space.sample())
      
def main(_):
  global env
  FLAGS.episode_len = int(FLAGS.episode_secs / FLAGS.light_secs)
  FLAGS.cars_per_sec = FLAGS.local_cars_per_sec * 3
  env = traffic_test.make_env(norender=True, randomized=False)
  profile_it()
  cProfile.run("profile_it()", "prof_bin")

if __name__ == '__main__':
  tf.app.run()
