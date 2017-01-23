import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
from gym_traffic.algorithms import *
import threading
import os.path
from functools import partial

flags.DEFINE_integer('threads', 4, 'Number of different threads to use')
flags.DEFINE_integer('a3c_batch', 30, 'Length of episode buffer')
flags.DEFINE_integer('report_rate', 5, 'Rate to print average reward')

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
    hidden = tl.fully_connected(self.flat_obs, num_outputs=150,
        normalizer_fn=tl.batch_norm, normalizer_params={'updates_collections': None})
    hidden2 = tl.fully_connected(hidden, num_outputs=150,
        normalizer_fn=tl.batch_norm, normalizer_params={'updates_collections': None})
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(150,state_is_tuple=True)
    self.c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
    self.h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
    self.c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], name="cin")
    self.h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name="hin")
    state_in = tf.nn.rnn_cell.LSTMStateTuple(self.c_in, self.h_in)
    lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
        lstm_cell, tf.expand_dims(hidden2, 0), initial_state=state_in)
    rnn_out = tf.reshape(lstm_outputs, [-1, 150])
    self.score = tl.fully_connected(rnn_out, num_outputs=self.num_actions, activation_fn=None)
    self.probs = tf.nn.sigmoid(self.score)
    self.value = tl.fully_connected(rnn_out, num_outputs=self.num_actions, activation_fn=None)
  
  def make_train_ops(self):
    self.update_local = update_target_graph('global', self.name)
    self.target_v = tf.placeholder(tf.float32, [None, self.num_actions], name="target_v")
    self.input_y = tf.placeholder(tf.float32, [None, self.num_actions], name="actions")
    self.advantages = tf.placeholder(tf.float32, [None, self.num_actions], name="advantages")
    policy_loss = tf.reduce_mean(self.advantages *
        tf.nn.sigmoid_cross_entropy_with_logits(self.score, self.input_y))
    value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - self.value))
    entropy = - tf.reduce_sum(self.probs * tf.log(self.probs))
    loss = 0.5 * value_loss + policy_loss - entropy * 0.01
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("entropy", entropy)
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

def validate(net, env, sess):
  reward_sum = 0
  obs = env.reset()
  rnn_state = [net.c_init, net.h_init]
  for i in range(FLAGS.episode_len):
    dist,rnn_state = sess.run([net.probs,net.lstm_state], feed_dict={net.observations: [obs],
      net.c_in: rnn_state[0], net.h_in: rnn_state[1]})
    if FLAGS.render: print("Action", dist)
    y, = np.round(dist)
    obs, reward, done, _ = env.step(y if net.vector_action else y[0])
    reward_sum += np.sum(reward)
    if done: break
  return reward_sum

def run(env_f):
  hack = env_f(norender=True) # We need to run an env first to compile it, because jit compilation isn't thread safe
  hack.reset()
  hack.step(hack.action_space.sample())
  with tf.device("/cpu:0"): 
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
    net.advantages: advantages, net.target_v: drs[:-1],
    net.c_in: net.c_init, net.h_in: net.h_init}
  if summary is not None: return sess.run([net.apply_grads,summary], feed_dict=fd)[1]
  else: return sess.run(net.apply_grads, feed_dict=fd)

def work(net, sess, save):
  writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, net.name))
  ys = np.empty((FLAGS.a3c_batch, net.num_actions), dtype=np.int32)
  vals = np.empty((FLAGS.a3c_batch + 1, net.num_actions), dtype=np.float32)
  xs = np.empty((FLAGS.a3c_batch, *net.env.observation_space.shape), dtype=np.float32)
  drs = np.empty((FLAGS.a3c_batch + 1, net.num_actions), dtype=np.float32)
  episode_rewards = np.zeros(FLAGS.report_rate, dtype=np.float32)
  epsilon = 1
  end_epsilon = np.random.uniform(0.05, 0.2)
  print("Started worker", net.name, "with target epsilon", end_epsilon)
  explore = globals()[FLAGS.exploration]

  for e in range(FLAGS.total_episodes):
    sess.run(net.update_local)
    obs = net.env.reset()
    episode_reward = 0
    rnn_state = [net.c_init, net.h_init]
    for mt in range(FLAGS.episode_len):
      t = mt % FLAGS.a3c_batch
      tfprob,v,rnn_state = sess.run([net.probs,net.value,net.lstm_state], feed_dict={
          net.c_in: rnn_state[0], net.h_in: rnn_state[1], net.observations:[obs]})
      y = explore(tfprob[0], epsilon)
      ys[t] = y.astype(np.float32)
      xs[t] = obs
      vals[t] = v[0]
      obs, reward, done, _ = net.env.step(y if net.vector_action else y[0])
      drs[t] = reward
      episode_reward += np.sum(reward)
      if t == FLAGS.a3c_batch - 1 and not done:
        vals[-1] = sess.run(net.value, feed_dict={net.observations: [obs],
          net.c_in: rnn_state[0], net.h_in: rnn_state[1]})[0]
        train(sess, net, None, xs, ys, vals, drs)
        sess.run(net.update_local)
      if done: break
    if t != FLAGS.a3c_batch - 1 or done:
      vals[t+1] = 0 if done else sess.run(net.value, feed_dict={
        net.observations: [obs], net.c_in: rnn_state[0], net.h_in: rnn_state[1]})[0,0]
      s = train(sess, net, net.summary, xs[:t+1], ys[:t+1], vals[:t+2], drs[:t+2])
      writer.add_summary(s, e)
      epsilon -= (epsilon - end_epsilon) / (FLAGS.total_episodes - e)

    episode_rewards[e % FLAGS.report_rate] = episode_reward
    if e % FLAGS.report_rate == FLAGS.report_rate - 1:
      reward_mean = np.mean(episode_rewards)
      print("Reward mean", reward_mean)
      s = sess.run(net.avg_summary, feed_dict={net.avg_r:reward_mean})
      writer.add_summary(s, e)

    if ((e % FLAGS.save_rate) == 0 or e == FLAGS.total_episodes - 1) \
        and threading.current_thread() == threading.main_thread():
      print("Saving")
      save()
