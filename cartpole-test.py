import gym
import tensorflow as tf
import gym_traffic.algorithms.polgrad as polgrad
import gym_traffic.algorithms.a3c as a3c
import gym_traffic.algorithms.a3c_rnn as a3c_rnn
import gym_traffic.algorithms.cem as cem

flags = tf.app.flags
FLAGS = flags.FLAGS

def main(_):
  FLAGS.learning_rate = 0.1
  FLAGS.batch_size = 10
  FLAGS.gamma = 0.99
  FLAGS.lam = 1
  globals()[FLAGS.trainer].run(lambda **_: gym.make('CartPole-v0'))

if __name__ == '__main__':
  tf.app.run()

