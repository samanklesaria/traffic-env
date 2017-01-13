import gym
import tensorflow as tf
import gym_traffic.algorithms.polgrad as trainer
# import gym_traffic.algorithms.a3c as trainer
# import gym_traffic.algorithms.cem as trainer

def main(_):
  trainer.run(lambda: gym.make('CartPole-v0'))

if __name__ == '__main__':
  tf.app.run()

