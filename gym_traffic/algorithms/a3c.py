import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as tl
import tensorflow.contrib.rnn as rnn
from gym_traffic.algorithms.util import *
import threading
import os.path
from functools import partial
from util import print_running_stats

EPS = 1e-8

# Copy one set of variables to another
def update_target_graph(from_scope, to_scope):
  return [dst.assign(src) for src, dst in zip(
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope),
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope))]

# Contains the network and environment for a single thread
class A3CNet:
  def __init__(self, env_f):
    self.name = tf.get_variable_scope().name
    self.env = env_f()
    obs_shape = self.env.observation_space.shape
    self.episode_num = tf.Variable(0,dtype=tf.int32,name='episode_num',trainable=False)
    self.increment_episode = tf.stop_gradient(self.episode_num.assign_add(1))
    self.observations = tf.placeholder(tf.float32, [None,*obs_shape], name="observations")
    gru = rnn.GRUCell(60)
    self.state_in = gru.zero_state(1, tf.float32)
    self.rnn_out, self.state_out = tf.nn.dynamic_rnn(gru,
        tf.expand_dims(self.observations,0),
        initial_state=self.state_in, dtype=np.float32)
    mid = tf.squeeze(self.rnn_out, 0, name="mid")
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, mid)
    self.hidden = tl.fully_connected(mid, 60, outputs_collections=tf.GraphKeys.ACTIVATIONS)
    self.score = tl.fully_connected(self.hidden, self.env.action_space.size, activation_fn=None)
    self.probs = tf.nn.sigmoid(self.score)
    tf.summary.histogram("probs", self.probs)
    self.value = tl.fully_connected(self.hidden, num_outputs=self.env.reward_size, activation_fn=None)

  def make_train_ops(self):
    self.update_local = update_target_graph('global', self.name)
    self.target_v = tf.placeholder(tf.float32, [None, self.env.reward_size], name="target_v")
    self.input_y = tf.placeholder(tf.float32, [None, self.env.action_space.size], name="actions")
    self.advantages = tf.placeholder(tf.float32, [None, self.env.reward_size], name="advantages")
    policy_loss = tf.reduce_sum(
        self.advantages * tf.nn.sigmoid_cross_entropy_with_logits(logits=self.score, labels=self.input_y))
    value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - self.value))
    entropy = -tf.reduce_mean(self.probs * tf.log(self.probs + EPS))
    loss = 0.5 * value_loss + policy_loss - entropy * 0.001
    tf.summary.scalar("mean_value", tf.mean(self.value))
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("entropy", entropy)
    tf.summary.scalar("value_loss", value_loss)
    tf.summary.scalar("policy_loss", policy_loss)
    self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    gradients = tf.gradients(loss,self.local_vars)
    self.grads,_ = tf.clip_by_global_norm(gradients,40.0)
    if FLAGS.grad_summary:
      prob_grads = [tf.gradients(self.probs[0][i],
        [self.observations, self.state_in]) for i in range(self.env.action_space.size)]
      for (i,g) in enumerate(prob_grads):
        tf.summary.histogram("obs_grad"+str(i), g[0])
        tf.summary.histogram("state_grad"+str(i), g[1])
    tl.summarize_activations(self.name)
    self.summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, self.name))

  def make_apply_ops(self, opt):
    self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    self.apply_grads = opt.apply_gradients(zip(self.grads,self.global_vars))

def make_worker(name, env_f):
  with tf.variable_scope(name):
    w = A3CNet(env_f)
    w.make_train_ops()
  return w

# Generator to simulate an epoch
def epoch(net, sess, extra_ops=[]):
  rnn = None
  obs = net.env.reset()
  ops = [net.probs, net.state_out, *extra_ops]
  for mt in range(FLAGS.episode_len):
    fd = {net.observations: [obs]}
    if rnn is not None: fd[net.state_in] = rnn
    dist,rnn,*others = sess.run(ops, feed_dict=fd)
    flaty = proportional(dist[0], None)
    y = net.env.action_space.to_action(flaty)
    new_obs, reward, done, _ = net.env.step(y)
    yield mt,obs,rnn,dist[0],flaty.astype(np.float32),reward,new_obs,done,others
    obs = new_obs
    if done: break

def validate(net, sess, writer):
  reward_sum = 0
  multiplier = 1.0
  for (i,obs,rnn,d,_,r,_,_,others) in epoch(net, sess):
    if FLAGS.render:
      print("Obs", obs)
      print("Hidden", rnn)
      print("Action", d)
    reward_sum += np.mean(r) * (multiplier if FLAGS.print_discounted else 1)
    multiplier *= FLAGS.gamma
  return reward_sum

# Run A3C training method
def train_model(net, sess, writer, weight_saver, workers):
  ev = threading.Event()
  coord = tf.train.Coordinator()
  threads = [threading.Thread(target=work, args=[w,sess,coord,ev]) for w in workers]
  model_file = os.path.join(FLAGS.logdir, 'model.ckpt')
  for t in threads: t.start()
  episode_num = None
  try:
    while not coord.should_stop():
      episode_num = sess.run(net.episode_num)
      sess.run(net.increment_episode)
      ev.wait()
      if (episode_num % FLAGS.summary_rate) == 0:
        rew = validate(net, sess, writer)
        print("Reward", rew)
        s = sess.run(net.avg_summary, feed_dict={net.avg_r:rew})
        writer.add_summary(s, episode_num)
      if (episode_num % FLAGS.save_rate) == 0:
        weight_saver.save(sess, model_file, global_step=episode_num)
      ev.clear()
  except:
    print("Waiting for threads to stop")
    coord.request_stop()
    coord.join(threads)
    print("Saving")
    if episode_num: weight_saver.save(sess, model_file, global_step=episode_num)
    raise

def run(env_f):
  if not FLAGS.restore: remkdir(FLAGS.logdir)
  with tf.device("/cpu:0"):
    with tf.variable_scope('global'):
      master = A3CNet(env_f)
      master.avg_r = tf.placeholder(tf.float32, name="avg_r")
      master.avg_summary = tf.summary.scalar("avg_r_summary", master.avg_r)
    if FLAGS.mode == "train":
      opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      workers = [make_worker('w'+str(t), env_f) for t in range(FLAGS.threads)]
      with tf.variable_scope('application'):
        for w in workers: w.make_apply_ops(opt)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      weight_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'),
          max_to_keep=10)
      if FLAGS.restore:
        latest = tf.train.latest_checkpoint(FLAGS.logdir)
        print("Restoring from", latest)
        weight_saver.restore(sess, latest)
        # WHAT ABOUT RESTORING THE WHOLE THING?
        # tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
        # we have to use stuff like tf.get_collection. Can't rely on object properties
        # filename = ".".join([tf.latest_checkpoint(train_dir), "meta"])
        # hyperparams should be saved as tf.get_collection("hparams")
      if FLAGS.mode == "validate":
        summary_writer = tf.summary.FileWriter("val_dir", tf.get_default_graph())
        def rewards():
          while True:
            yield validate(master, sess, summary_writer)
        print_running_stats(rewards())
      elif FLAGS.mode == "weights":
        import json
        local_vars = tf.trainable_variables()
        local_vals = sess.run(local_vars)
        staging = {}
        for (v, val) in zip(local_vars, local_vals):
          staging[v.name] = val.tolist()
        with open('weights.json', 'w') as f:
          json.dump(staging, f, indent=4, separators=(',',':'))
      elif FLAGS.mode == "train":
        summary_writer = tf.summary.FileWriter(FLAGS.logdir, tf.get_default_graph())
        train_model(master, sess, summary_writer, weight_saver, workers)
      else: print("Unknown mode", FLAGS.mode)

# Running training step on previous epoch, using generalized advantage
def train(sess, net, summary, xs, ys, vals, drs):
  drs[-1] = vals[-1]
  advantages = drs[:-1] + FLAGS.gamma * vals[1:] - vals[:-1]
  drs = discount(drs, FLAGS.lam * FLAGS.gamma)
  advantages = discount(advantages, FLAGS.gamma)
  fd = {net.observations: xs, net.input_y: ys,
    net.advantages: advantages, net.target_v: drs[:-1]}
  return sess.run([net.apply_grads,summary], feed_dict=fd)[1]

# Run an A3C worker thread
def work(net, sess, coord, ev):
  writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, net.name))
  ys = np.empty((FLAGS.batch_size, net.env.action_space.size), dtype=np.float32)
  vals = np.empty((FLAGS.batch_size + 1, net.env.reward_size), dtype=np.float32)
  xs = np.empty((FLAGS.batch_size, *net.env.observation_space.shape), dtype=np.float32)
  drs = np.empty((FLAGS.batch_size + 1, net.env.reward_size), dtype=np.float32)
  episode_rewards = np.zeros(FLAGS.summary_rate, dtype=np.float32)
  print("Started worker", net.name)
  # with coord.stop_on_exception():
  while not coord.should_stop():
    episode_num = sess.run(net.episode_num)
    sess.run(net.increment_episode)
    sess.run(net.update_local)
    for (mt,obs,_,_,y,reward,new_obs,done,v) in epoch(net, sess, [net.value]):
      t = mt % FLAGS.batch_size
      ys[t] = y
      xs[t] = obs
      vals[t] = v[0][0]
      drs[t] = reward / 100.0
      if t == FLAGS.batch_size - 1 and not done:
        vals[-1] = sess.run(net.value, feed_dict={net.observations: [new_obs]})[0]
        s = train(sess, net, net.summary, xs, ys, vals, drs)
        sess.run(net.update_local)
      if done: break
    if t != FLAGS.batch_size - 1 or done:
      vals[t+1] = 0 if done else sess.run(net.value, feed_dict={
          net.observations: [obs]})[0,0]
      s = train(sess, net, net.summary, xs[:t+1], ys[:t+1], vals[:t+2], drs[:t+2])
    writer.add_summary(s, episode_num)
    ev.set()
