import gym
import tensorflow as tf
import gym_traffic.algorithms.polgrad as polgrad
# import gym_traffic.algorithms.a3c as a3c
# import gym_traffic.algorithms.cem as cem

def main(_):
  polgrad.run(lambda: gym.make('CartPole-v0'))

if __name__ == '__main__':
  tf.app.run()

