import tensorflow as tf
import scipy.signal
import numpy as np
import os.path
import os
from functools import partial

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('restore', False, "Should we restore from checkpoint")
flags.DEFINE_boolean('grad_summary', False, "Should we show obs grad summaries")
flags.DEFINE_boolean('print_discounted', True, "Should we print total episode rewards discounted")
flags.DEFINE_boolean('save_settings', False, "Should we save settings")
flags.DEFINE_boolean('validate', False, 'Run a validation loop without training')
flags.DEFINE_boolean('render', False, 'Render during validation')
flags.DEFINE_boolean('obs_pic', False, 'Show a picture of the observations')
flags.DEFINE_boolean('print_grad', False, 'Print the gradient on validation')
flags.DEFINE_integer('episode_len', 500, 'Number of actions per episode')
flags.DEFINE_integer('total_episodes', 10000, 'Total number of episodes to train')
flags.DEFINE_integer('save_rate', 1000, 'Write weights to disk every save_rate grad calculations')
flags.DEFINE_string('logdir', 'summaries', 'Log directory')
flags.DEFINE_float('gamma', 0.99, 'Discount factor')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('summary_rate', 25, 'Show summary every how many episodes')
flags.DEFINE_string('trainer', "a3c", 'Training algorithm to use')
flags.DEFINE_string('exploration', "proportional", 'Exploration strategy to use')
flags.DEFINE_integer('batch_size', 60, 'Update params every how many episodes')
flags.DEFINE_float('lam', 1, 'Lambda used in Generalized Advantage Estimation')
flags.DEFINE_integer('threads', 4, 'Number of different threads to use')
flags.DEFINE_integer('vis_size', 200, "Number of observations to plot with tsne")
flags.DEFINE_string('mode', 'train', "Options are 'weights', 'train', 'validate', 'embed'")

def remkdir(d):
  if tf.gfile.Exists(d): tf.gfile.DeleteRecursively(d)
  tf.gfile.MakeDirs(d)

class in_dir:
  def __init__(self, dirname): self.dirname = dirname
  def __enter__(self):
    remkdir(self.dirname)
    os.chdir(self.dirname)
  def __exit__(self, *_):
    os.chdir("..")

def e_greedy(probs, epsilon):
  return np.where(np.random.uniform(size=probs.shape) < epsilon,
      np.round(np.random.uniform(size=probs.shape)),
      np.round(probs)).astype(np.int)

def proportional(probs, epsilon):
  return (np.random.uniform(size=probs.shape) < probs).astype(np.int8)

def centered(probs, epsilon):
  shifted = (epsilon * 0.5 + (1 - epsilon) * probs) / 2
  return (np.random.uniform(size=shifted.shape) < shifted).astype(np.int)

# Disount future rewards
def discount(a, gamma):
  return scipy.signal.lfilter([1], [1, -gamma], a[::-1], axis=0)[::-1]
