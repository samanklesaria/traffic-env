import gym
import numpy as np
import tensorflow as tf
import gym_traffic.algorithms.polgrad as polgrad
import gym_traffic.algorithms.a3c as a3c
import gym_traffic.algorithms.cem as cem
import gym_traffic.algorithms.random as random
import gym.spaces as spaces

flags = tf.app.flags
FLAGS = flags.FLAGS

def prepro(frame):
  frame = frame[35:195] # crop
  frame = frame[::2,::2,0] # downsample by factor of 2
  frame[frame == 144] = 0 # erase background (background type 1)
  frame[frame == 109] = 0 # erase background (background type 2)
  frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
  return frame.astype(np.float).ravel()

def make_env(norender=False, **_):
  env = gym.make('Pong-v0')
  if FLAGS.render and not norender: env.rendering = True
  return ActionTranslate(Preprocessor(env))

class Preprocessor(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.prev_x = None
    self.observation_space = spaces.Box(0, 255, (6400,))

  def _observation(self, obs):
    cur_x = prepro(obs)
    x = cur_x - self.prev_x if self.prev_x is not None else np.zeros_like(cur_x)
    self.prev_x = cur_x
    return x

class ActionTranslate(gym.ActionWrapper):
  def _action(self, action):
    return 2 if action else 3

def main(_):
  FLAGS.gamma = 0.99
  globals()[FLAGS.trainer].run(make_env)

if __name__ == '__main__':
  tf.app.run()

