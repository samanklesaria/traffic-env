import traffic_test
import tensorflow as tf
from gym_traffic.algorithms import in_dir

flags = tf.app.flags
FLAGS = flags.FLAGS

def main(_):
  with in_dir("pretraining"):
    FLAGS.change_penality = 1
    FLAGS.local_weight = 1
    FLAGS.reward_counts = 0
    FLAGS.total_episodes = 500
    traffic_test.main()
  with in_dir("training"):
    FLAGS.local_weight = 3
    FLAGS.change_penality = 1
    FLAGS.reward_counts = 1
    FLAGS.total_episodes = 2000
    FLAGS.restore = True
    FLAGS.checkpoints = "../pretraining/checkpoints"
    traffic_test.main()

if __name__ == '__main__':
  tf.app.run()
