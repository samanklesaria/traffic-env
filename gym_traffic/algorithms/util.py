import tensorflow as tf
import scipy.signal
import numpy as np
import os.path
import os
from functools import partial
from args import add_argument, add_derivation, FLAGS, PARSER
from util import *
import json

add_argument('--restore', False, type=bool)
add_argument('--grad_summary', False, type=bool)
add_argument('--reward_printing', 'discounted')
add_argument('--render', False, type=bool)
add_argument('--episode_len', 500, type=int)
add_argument('--save_rate', 1000, type=int)
add_argument('--logdir', 'summaries')
add_argument('--gamma', 0.99, type=float)
add_argument('--learning_rate', 0.001, type=float)
add_argument('--summary_rate', 5, type=int)
add_argument('--validate_rate', 25, type=int)
add_argument('--trainer', "qlearn")
add_argument('--exploration', "e_greedy")
add_argument('--batch_size', 30, type=int)
add_argument('--vis_size', 200, type=int)
add_argument('--mode', 'train')
add_argument('--spacing', 3, type=int)
add_argument('--start_eps', 0.8, type=float)
add_argument('--end_eps', 0.01, type=float)
add_argument('--start_temp', 0.2, type=float)
add_argument('--end_temp', 0.01, type=float)
add_argument('--annealing_episodes', 10000, type=float)
add_argument('--history', 1, type=int)
add_argument('--target_update_rate', 50, type=int)
add_argument('--buffer_size', 1000, type=int)
add_argument('--trace_size', 8, type=int)
add_argument('--threads', 4, type=int)
add_argument('--lam', 1, type=float)
add_argument('--debug', False, type=bool)
add_argument('--train_rate', 1, type=int)

def standard_derivations():
  FLAGS.print_discounted = FLAGS.reward_printing == "discounted"
  FLAGS.print_avg = FLAGS.reward_printing == "avg"
  if FLAGS.render: FLAGS.threads = 1
add_derivation(standard_derivations)

EPS = 1e-8

def entropy(probs):
  tf.summary.histogram("probs", probs)
  s = tf.negative(tf.reduce_mean(probs * tf.log(probs + EPS)), name="entropy")
  tf.summary.scalar("entropy_val", s)

def anneal(varname, start, end):
  var = tf.Variable(start, dtype=tf.float32,name=varname,trainable=False)
  tf.assign(var, tf.maximum(end, var -
    ((start - end) / FLAGS.annealing_episodes)), name=("dec_" + varname))
  tf.summary.scalar(varname+"_val", var)
  return var

def ref(tensor_name):
  return tf.get_default_graph().get_tensor_by_name(tensor_name)

def handle_modes(env_f, model, val, train):
  if not FLAGS.restore:
    remkdir(FLAGS.logdir)
    with open(os.path.join(FLAGS.logdir, "settings.json"),'w') as f:
      PARSER.defaults.update(FLAGS.__dict__)
      json.dump(PARSER.defaults, f, indent=4,separators=(',',': '))
    env = env_f()
    model(env)
    tf.summary.scalar("avg_r_summary", tf.placeholder(tf.float32, name="avg_r"))
    init = tf.global_variables_initializer()
  with tf.Session() as sess:
    if FLAGS.restore:
      with open(os.path.join(FLAGS.logdir, "settings.json"),'r') as f:
        PARSER.defaults.update(json.load(f))
      latest = tf.train.latest_checkpoint(FLAGS.logdir)
      tf.train.import_meta_graph(latest + '.meta').restore(sess, latest)
    else:
      env = env_f()
      sess.run(init)
    if FLAGS.mode == "validate":
      print_running_stats(forever(lambda: val(sess, env)))
    elif FLAGS.mode == "train":
      if FLAGS.restore: summary_writer = tf.summary.FileWriter(FLAGS.logdir)
      else: summary_writer = tf.summary.FileWriter(FLAGS.logdir, tf.get_default_graph())
      saver = tf.train.Saver()
      model_file = os.path.join(FLAGS.logdir, 'model.ckpt')
      if FLAGS.debug:
        from tensorflow.python import debug as tf_debug
        dbg = tf_debug.LocalCLIDebugWrapperSession(sess)
        # dbg.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
      else: dbg = None
      train(sess, dbg, summary_writer, partial(saver.save, sess, model_file), env)

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

def exploration_param():
  if FLAGS.exploration == "boltzman":
    eps = anneal("temp", FLAGS.start_temp, FLAGS.end_temp)
  else:
    eps = anneal("eps", FLAGS.start_eps, FLAGS.end_eps)
  return eps

# does boltzman work for single agent too?

def softmax_decision(scores, eps):
  tf.summary.histogram("scores", scores)
  greedy = tf.cast(tf.argmax(scores, axis=-1, name="greedy"), tf.int32)
  if FLAGS.exploration == "boltzman":
    heated_scores = scores / eps
    entropy(tf.nn.softmax(scores))
    tf.map_fn(lambda r: tf.squeeze(tf.multinomial(r, 1), 1),
        heated_scores, back_prop=False, name="explore")
  elif FLAGS.exploration == "e_greedy":
    score_shape = tf.shape(scores)
    num_actions = score_shape[-1]
    rand = tf.random_uniform(tf.shape(greedy), maxval=num_actions, dtype=tf.int32)
    condition = tf.random_uniform(score_shape[:-1]) < eps
    tf.where(condition, rand, greedy, name="explore")
  else:
    raise Exception("Unknown exploration type " + FLAGS.exploration)

def sigmoid_decision(scores, eps):
  tf.summary.histogram("scores", scores)
  probs = tf.nn.sigmoid(scores)
  entropy(probs)
  tf.cast(tf.round(probs), tf.int32, name="greedy")
  if FLAGS.exploration == "e_greedy":
    shifted = eps * 0.5 + (1 - eps) * probs
  elif FLAGS.exploration == "proportional":
    shifted = probs
  else:
    raise Exception("Unknown exploration type " + FLAGS.exploration)
  tf.less(tf.random_uniform(tf.shape(shifted)), shifted, name="explore")

def isclose(a,b):
  assert np.all(np.abs(a - b) < 0.0001), (a, b)

def discount(a, gamma):
  return scipy.signal.lfilter([1], [1, -gamma], a[::-1], axis=0)[::-1]
