import tensorflow as tf
import scipy.signal
import numpy as np
import os.path
import os
from functools import partial

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoints', 'checkpoints', 'Checkpoint directory')
flags.DEFINE_boolean('weights', False, "Should we just print the saved weights?")
flags.DEFINE_boolean('restore', False, "Should we restore from checkpoint")
flags.DEFINE_boolean('print_discounted', True, "Should we print total episode rewards discounted")
flags.DEFINE_boolean('save_settings', False, "Should we save settings")
flags.DEFINE_boolean('validate', False, 'Run a validation loop without training')
flags.DEFINE_boolean('render', False, 'Render during validation')
flags.DEFINE_boolean('obs_pic', False, 'Show a picture of the observations')
flags.DEFINE_boolean('cnn', True, 'Use a hidden cnn')
flags.DEFINE_boolean('print_grad', False, 'Print the gradient on validation')
flags.DEFINE_integer('episode_len', 500, 'Number of actions per episode')
flags.DEFINE_integer('total_episodes', 10000, 'Total number of episodes to train')
flags.DEFINE_integer('save_rate', 400, 'Update params every how many batches')
flags.DEFINE_string('logdir', 'summaries', 'Log directory')
flags.DEFINE_float('gamma', 0.97, 'Discount factor')
flags.DEFINE_float('learning_rate', 0.0003, 'Learning rate')
flags.DEFINE_integer('summary_rate', 10, 'Show summary every how many episodes')
flags.DEFINE_integer('validate_rate', 5, 'Validate every how many episodes')
flags.DEFINE_string('trainer', "a3c", 'Training algorithm to use')
flags.DEFINE_string('exploration', "proportional", 'Exploration strategy to use')
flags.DEFINE_integer('batch_size', 30, 'Update params every how many episodes')
flags.DEFINE_float('lam', 0.98, 'Lambda used in Generalized Advantage Estimation')
flags.DEFINE_integer('threads', 4, 'Number of different threads to use')
flags.DEFINE_integer('seed_len', 200, 'Iterations before choosing a new seed')

class in_dir:
  def __init__(self, dirname): self.dirname = dirname
  def __enter__(self):
    if tf.gfile.Exists(self.dirname):
      tf.gfile.DeleteRecursively(self.dirname)
    tf.gfile.MakeDirs(self.dirname)
    os.chdir(self.dirname)
  def __exit__(self, *_):
    os.chdir("..")

def add_rl_vars(self, env):
  self.episode_num = tf.Variable(0,dtype=tf.int32,name='episode_num',trainable=False)
  self.increment_episode = tf.stop_gradient(self.episode_num.assign_add(1))
  self.observations = tf.placeholder(tf.float32, [None,*env.observation_space.shape], name="input_x")

def load_from_checkpoint(sess, var_list=None):
  tf.gfile.MakeDirs(FLAGS.checkpoints)
  checkpoint_file = os.path.join(FLAGS.checkpoints, "model.ckpt")
  saver = tf.train.Saver(var_list, keep_checkpoint_every_n_hours=1)
  if FLAGS.restore:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restoring from", ckpt.model_checkpoint_path)
  else: print("Not restoring")
  return partial(saver.save, sess, checkpoint_file)

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
