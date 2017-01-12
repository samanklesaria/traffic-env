import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
from gym_traffic.algorithms import *
import threading
import os.path
from functools import partial

flags.DEFINE_integer('threads', 4, 'Number of different threads to use')

# Copy one set of variables to another
def update_target_graph(from_scope, to_scope):
  return [dst.assign(src) for src, dst in zip(
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope),
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope))]

# Contains the network and environment for a single thread
class A3CNet(TFAgent):
  def __init__(self, env_f):
    self.name = tf.get_variable_scope().name
    self.env = env_f()
    super().__init__(self.env)
    hidden = tl.fully_connected(self.flat_obs, num_outputs=100)
    hidden2 = tl.fully_connected(hidden, num_outputs=80)
    hidden3 = tl.fully_connected(hidden, num_outputs=100)
    self.score = tl.fully_connected(hidden3, num_outputs=self.num_actions, activation_fn=None)
    self.probs = tf.nn.sigmoid(self.score)
    self.value = tl.fully_connected(hidden3, num_outputs=self.num_actions, activation_fn=None)
  
  def make_train_ops(self):
    self.update_local = update_target_graph('global', self.name)
    self.target_v = tf.placeholder(tf.float32, [None, self.num_actions], name="target_v")
    self.input_y = tf.placeholder(tf.float32, [None, self.num_actions], name="actions")
    self.advantages = tf.placeholder(tf.float32, [None, self.num_actions], name="advantages")
    policy_loss = tf.reduce_mean(self.advantages *
        tf.nn.sigmoid_cross_entropy_with_logits(self.score, self.input_y))
    value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - self.value))
    self.entropy = - tf.reduce_sum(self.probs * tf.log(self.probs))
    self.step = tf.placeholder(tf.int32, name="step")
    # start by encouraging exploration, gradually encourage discrete choices
    # explorer = tf.tanh(self.step - FLAGS.total_episodes / 2) 
    loss = 0.5 * value_loss + policy_loss
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("value_loss", value_loss)
    tf.summary.scalar("policy_loss", policy_loss)
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    gradients = tf.gradients(loss,local_vars)
    self.grads, grad_norms = tf.clip_by_global_norm(gradients,40.0)
    tf.summary.scalar("grad_norm", grad_norms)
    self.summary = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, self.name))
    self.avg_r = tf.placeholder(tf.float32, name="avg_r")
    self.avg_summary = tf.summary.scalar("avg_r_summary", self.avg_r)

  def make_apply_ops(self, opt):
    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    self.apply_grads = opt.apply_gradients(zip(self.grads,global_vars))

def make_worker(name, env_f):
  with tf.variable_scope(name):
    w = A3CNet(env_f)
    w.make_train_ops()
  return w

def run(env_f):
  with tf.variable_scope('global'): master = A3CNet(env_f)
  if not FLAGS.validate: 
    if tf.gfile.Exists(FLAGS.logdir):
      tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.gfile.MakeDirs(FLAGS.logdir)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    workers = [make_worker('w'+str(t), env_f) for t in range(FLAGS.threads)]
    with tf.variable_scope('application'):
      for w in workers: w.make_apply_ops(opt)
    gw = tf.summary.FileWriter(os.path.join(FLAGS.logdir, "graph"),
        tf.get_default_graph())
    gw.close()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    master.load_from_checkpoint(sess,
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'))
    if FLAGS.validate:
      while True:
        print(validate(master, master.env, sess))
    else:
      threads = [threading.Thread(target=work, args=[w, sess, None]) for w in workers[1:]]
      for t in threads: t.start()
      work(workers[0], sess, partial(master.saver.save, sess, master.checkpoint_file))

def train(sess, net, summary, xs, ys, vals, drs):
  drs[-1] = vals[-1]
  advantages = drs[:-1] + FLAGS.gamma * vals[1:] - vals[:-1]
  discount(drs, FLAGS.gamma)
  discount(advantages, FLAGS.gamma)
  fd = {net.observations: xs, net.input_y: ys,
    net.advantages: advantages, net.target_v: drs[:-1]}
  if summary is not None: return sess.run([net.apply_grads,summary], feed_dict=fd)[1]
  else: return sess.run(net.apply_grads, feed_dict=fd)

def work(net, sess, save):
  writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, net.name))
  ys = np.empty((30, net.num_actions), dtype=np.float32)
  vals = np.empty((31, net.num_actions), dtype=np.float32)
  xs = np.empty((30, *net.env.observation_space.shape), dtype=np.float32)
  drs = np.empty((31, net.num_actions), dtype=np.float32)
  print("Started worker", net.name)
  episode_rewards = np.zeros(5, dtype=np.float32)

  for e in range(FLAGS.total_episodes):
    sess.run(net.update_local)
    obs = net.env.reset()
    episode_reward = 0
    for mt in range(FLAGS.episode_len):
      t = mt % 30
      tfprob, v = sess.run([net.probs,net.value], feed_dict={
          net.observations:[obs]})
      y = (np.random.uniform(size=tfprob[0].shape) < tfprob[0]).astype(np.int8)
      ys[t] = y.astype(np.float32)
      xs[t] = obs
      vals[t] = v[0]
      obs, reward, done, _ = net.env.step(y if net.vector_action else y[0])
      drs[t] = reward / 1000.0
      episode_reward += np.sum(reward)
      if t == 29 and not done:
        vals[30] = sess.run(net.value, feed_dict={net.observations: [obs]})[0]
        train(sess, net, None, xs, ys, vals, drs)
        sess.run(net.update_local)
      if done: break
    if t != 29 or done:
      vals[t+1] = 0 if done else sess.run(net.value, feed_dict={net.observations: [obs]})[0,0]
      s = train(sess, net, net.summary, xs[:t+1], ys[:t+1], vals[:t+2], drs[:t+2])
      writer.add_summary(s, e)

    episode_rewards[e % 5] = episode_reward
    if e % 5 == 4:
      reward_mean = np.mean(episode_rewards)
      print("Reward mean", reward_mean)
      s = sess.run(net.avg_summary, feed_dict={net.avg_r:reward_mean})
      writer.add_summary(s, e)

    if (e % FLAGS.save_rate) == 0 and threading.current_thread() == threading.main_thread():
      print("Saving")
      save()
