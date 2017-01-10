import tensorflow as tf
import numpy as np
import os.path
from numba import jit

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoints', 'checkpoints', 'Checkpoint directory')
flags.DEFINE_boolean('restore', False, "Should we restore from checkpoint")
flags.DEFINE_boolean('validate', False, 'Run a validation loop without training')
flags.DEFINE_integer('episode_len', 800, 'Number of actions per episode')
flags.DEFINE_integer('total_episodes', 1000, 'Total number of episodes to train')
flags.DEFINE_integer('save_rate', 100, 'Update params every how many batches')
flags.DEFINE_string('logdir', 'summaries', 'Log directory')
flags.DEFINE_float('gamma', 0.99, 'Discount factor')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_integer('summary_rate', 5, 'Show summary every how many episodes')
flags.DEFINE_integer('validate_rate', 5, 'Validate every how many episodes')

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
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("Restoring from", ckpt.model_checkpoint_path)

def validate(net, env, sess):
  reward_sum = 0
  obs = env.reset()
  for _ in range(FLAGS.episode_len):
    y, = np.round(sess.run(net.probs,feed_dict={net.observations: [obs]})).astype(np.int8)
    obs, reward, done, _ = env.step(y if net.vector_action else y[0])
    reward_sum += reward
    if done: break
  return reward_sum

def e_greedy(probs, epsilon):
  return np.where(np.random.uniform(size=probs.shape) < epsilon,
      np.round(np.random.uniform(size=probs.shape)),
      np.round(probs)).astype(np.int8)

# Disount future rewards
@jit("void(float32[:,:], float32)", nopython=True, nogil=True)
def discount(rewards, gamma):
  for i in range(rewards.shape[0]-1, 0, -1):
    rewards[i-1] = rewards[i-1] + rewards[i] * gamma
