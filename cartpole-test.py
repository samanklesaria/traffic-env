import gym
import tensorflow as tf
import gym_traffic.algorithms.polgrad as polgrad
import gym_traffic.algorithms.a3c as a3c
import gym_traffic.algorithms.cem as cem

flags = tf.app.flags
FLAGS = flags.FLAGS

def main(_):
  FLAGS.learning_rate = 0.01
  FLAGS.gamma = 0.99
  globals()[FLAGS.trainer].run(lambda **_: gym.make('CartPole-v0'))

if __name__ == '__main__':
  tf.app.run()

