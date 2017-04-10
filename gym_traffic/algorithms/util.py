import tensorflow as tf
import scipy.signal
import numpy as np
import os.path
import os
from functools import partial
from args import add_argument, FLAGS

add_argument('--restore', False, type=bool)
add_argument('--grad_summary', False, type=bool)
add_argument('--print_discounted', True, type=bool)
add_argument('--save_settings', False, type=bool)
add_argument('--validate', False, type=bool)
add_argument('--render', False, type=bool)
add_argument('--obs_pic', False, type=bool)
add_argument('--print_grad', False, type=bool)
add_argument('--episode_len', 500, type=int)
add_argument('--total_episodes', 10000, type=int)
add_argument('--save_rate', 1000, type=int)
add_argument('--logdir', 'summaries')
add_argument('--gamma', 0.99, type=float)
add_argument('--learning_rate', 0.0007, type=float)
add_argument('--summary_rate', 25, type=int)
add_argument('--trainer', "a3c")
add_argument('--exploration', "proportional")
add_argument('--batch_size', 60, type=int)
add_argument('--lam', 0.99, type=float)
add_argument('--threads', 4, type=int)
add_argument('--vis_size', 200, type=int)
add_argument('--mode', 'train')

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
