import tensorflow as tf
import numpy as np
import os.path
from numba import jit
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoints', 'checkpoints', 'Checkpoint directory')
flags.DEFINE_boolean('weights', False, "Should we just print the saved weights?")
flags.DEFINE_boolean('restore', False, "Should we restore from checkpoint")
flags.DEFINE_boolean('print_discounted', False, "Should we print total episode rewards discounted")
flags.DEFINE_boolean('save_settings', False, "Should we save settings")
flags.DEFINE_boolean('validate', False, 'Run a validation loop without training')
flags.DEFINE_boolean('render', False, 'Render during validation')
flags.DEFINE_integer('episode_len', 5000, 'Number of actions per episode')
flags.DEFINE_integer('total_episodes', 10000, 'Total number of episodes to train')
flags.DEFINE_integer('save_rate', 400, 'Update params every how many batches')
flags.DEFINE_string('logdir', 'summaries', 'Log directory')
flags.DEFINE_float('gamma', 0.99, 'Discount factor')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
flags.DEFINE_integer('summary_rate', 10, 'Show summary every how many episodes')
flags.DEFINE_integer('validate_rate', 5, 'Validate every how many episodes')
flags.DEFINE_string('trainer', "polgrad_conv", 'Training algorithm to use')
flags.DEFINE_string('exploration', "proportional", 'Exploration strategy to use')
flags.DEFINE_integer('batch_size', 30, 'Update params every how many episodes')
flags.DEFINE_float('lam', 0.97, 'Lambda used in Generalized Advantage Estimation')
flags.DEFINE_integer('threads', 4, 'Number of different threads to use')

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
    self.episode_num = tf.Variable(0,dtype=tf.int32,name='episode_num',trainable=False)
    self.increment_episode = tf.stop_gradient(self.episode_num.assign_add(1))
    self.observations = tf.placeholder(tf.float32, [None,*env.observation_space.shape], name="input_x")

  def load_from_checkpoint(self, sess, var_list=None):
    tf.gfile.MakeDirs(FLAGS.checkpoints)
    self.checkpoint_file = os.path.join(FLAGS.checkpoints, "model.ckpt")
    self.saver = tf.train.Saver(var_list, keep_checkpoint_every_n_hours=1)
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

def centered(probs, epsilon):
  shifted = (epsilon * 0.5 + (1 - epsilon) * probs) / 2
  return (np.random.uniform(size=shifted.shape) < shifted).astype(np.int)

# Disount future rewards
@jit("void(float32[:,:], float32)", nopython=True, nogil=True)
def discount(rewards, gamma):
  for i in range(rewards.shape[0]-1, 0, -1):
    rewards[i-1] = rewards[i-1] + rewards[i] * gamma
