import tensorflow as tf
import numpy as np
import os.path
from numba import jit
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoints', 'checkpoints', 'Checkpoint directory')
flags.DEFINE_boolean('restore', False, "Should we restore from checkpoint")
flags.DEFINE_boolean('restore_settings', False, "Should we restore old settings?")
flags.DEFINE_boolean('validate', False, 'Run a validation loop without training')
flags.DEFINE_boolean('render', False, 'Render during validation')
flags.DEFINE_integer('episode_len', 5000, 'Number of actions per episode')
flags.DEFINE_integer('total_episodes', 3000, 'Total number of episodes to train')
flags.DEFINE_integer('save_rate', 100, 'Update params every how many batches')
flags.DEFINE_string('logdir', 'summaries', 'Log directory')
flags.DEFINE_float('gamma', 0.99, 'Discount factor')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_integer('summary_rate', 5, 'Show summary every how many episodes')
flags.DEFINE_integer('validate_rate', 5, 'Validate every how many episodes')
flags.DEFINE_string('trainer', "a3c", 'Training algorithm to use')
flags.DEFINE_string('exploration', "proportional", 'Exploration strategy to use')

class in_dir:
  def __init__(self, dirname): self.dirname = dirname
  def __enter__(self):
    if tf.gfile.Exists(self.dirname):
      tf.gfile.DeleteRecursively(self.dirname)
    tf.gfile.MakeDirs(self.dirname)
    os.chdir(self.dirname)
  def __exit__(self, *_):
    os.chdir("..")

class TFAgent:
  def __init__(self, env):
    self.vector_action = True
    try: self.num_actions = env.action_space.shape
    except:
      self.num_actions = 1
      self.vector_action = False
    self.num_inputs = np.prod(env.observation_space.shape)
    self.observations = tf.placeholder(tf.float32, [None,*env.observation_space.shape], name="input_x")
    self.flat_obs = tf.reshape(self.observations, [-1, self.num_inputs])

  def load_from_checkpoint(self, sess, var_list=None):
    tf.gfile.MakeDirs(FLAGS.checkpoints)
    self.checkpoint_file = os.path.join(FLAGS.checkpoints, "model.ckpt")
    self.saver = tf.train.Saver(var_list)
    if FLAGS.restore:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints)
      self.saver.restore(sess, ckpt.model_checkpoint_path)
      print("Restoring from", ckpt.model_checkpoint_path)

def e_greedy(probs, epsilon):
  return np.where(np.random.uniform(size=probs.shape) < epsilon,
      np.round(np.random.uniform(size=probs.shape)),
      np.round(probs)).astype(np.int)

def proportional(probs, epsilon):
  return (np.random.uniform(size=probs.shape) < probs).astype(np.int)

# Disount future rewards
@jit("void(float32[:,:], float32)", nopython=True, nogil=True)
def discount(rewards, gamma):
  for i in range(rewards.shape[0]-1, 0, -1):
    rewards[i-1] = rewards[i-1] + rewards[i] * gamma
