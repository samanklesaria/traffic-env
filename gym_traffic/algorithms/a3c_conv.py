import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
from gym_traffic.algorithms import *
import threading
import os.path
from functools import partial

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
    channels = np.prod(self.env.observation_space.shape[:-2])
    dims = self.env.observation_space.shape[-2:]
    flat_obs = tf.reshape(self.observations, [-1, channels, *dims])
    nhwc = tf.transpose(flat_obs, perm=[0,2,3,1])
    local = tl.convolution(nhwc, 1, [1,1], biases_initializer=None, activation_fn=None)
    self.score = tf.reshape(local, [-1, self.num_actions])
    self.value = tl.fully_connected(tf.reshape(nhwc, [-1, self.num_inputs]),
        self.num_actions, activation_fn=None)
    self.probs = tf.nn.sigmoid(self.score)
  
  def make_train_ops(self):
    self.update_local = update_target_graph('global', self.name)
    self.target_v = tf.placeholder(tf.float32, [None, self.num_actions], name="target_v")
    self.input_y = tf.placeholder(tf.float32, [None, self.num_actions], name="actions")
    self.advantages = tf.placeholder(tf.float32, [None, self.num_actions], name="advantages")
    policy_loss = tf.reduce_sum(self.advantages *
        tf.nn.sigmoid_cross_entropy_with_logits(self.score, self.input_y))
    value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - self.value))
    entropy = -tf.reduce_mean(tf.reduce_sum(self.probs * tf.log(self.probs), axis=1))
    loss = 0.5 * value_loss + policy_loss - entropy * 0.001
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("entropy", entropy)
    tf.summary.scalar("value_loss", value_loss)
    tf.summary.scalar("policy_loss", policy_loss)
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    gradients = tf.gradients(loss,local_vars)
    self.grads = [tf.clip_by_value(g, -10, 10) for g in gradients]
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
  for _ in range(FLAGS.episode_len):
    dist, = sess.run(net.probs, feed_dict={net.observations: [obs]})
    if FLAGS.render: print("Action", dist)
    y = np.round(dist)
    obs, reward, done, _ = env.step(y if net.vector_action else y[0])
    reward_sum += np.sum(reward)
    if done: break
  return reward_sum

def run(env_f):
  hack = env_f(norender=True)
  hack.reset()
  hack.step(hack.action_space.sample())
  del hack
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
  discount(drs, FLAGS.lam * FLAGS.gamma)
  discount(advantages, FLAGS.gamma)
  fd = {net.observations: xs, net.input_y: ys,
    net.advantages: advantages, net.target_v: drs[:-1]}
  if summary is not None: return sess.run([net.apply_grads,summary], feed_dict=fd)[1]
  else: sess.run(net.apply_grads, feed_dict=fd)

def work(net, sess, save):
  writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, net.name))
  ys = np.empty((FLAGS.batch_size, net.num_actions), dtype=np.float32)
  vals = np.empty((FLAGS.batch_size + 1, net.num_actions), dtype=np.float32)
  xs = np.empty((FLAGS.batch_size, *net.env.observation_space.shape), dtype=np.float32)
  drs = np.empty((FLAGS.batch_size + 1, net.num_actions), dtype=np.float32)
  episode_rewards = np.zeros(FLAGS.summary_rate, dtype=np.float32)
  epsilon = 1
  end_epsilon = np.random.uniform(0.05, 0.2)
  print("Started worker", net.name, "with target epsilon", end_epsilon)
  explore = globals()[FLAGS.exploration]

  for e in range(FLAGS.total_episodes):
    episode_num = sess.run(net.episode_num)
    sess.run(net.increment_episode)
    try:
      sess.run(net.update_local)
      obs = net.env.reset()
      episode_reward = 0
      multiplier = 1.0
      for mt in range(FLAGS.episode_len):
        t = mt % FLAGS.batch_size
        tfprob,v = sess.run([net.probs,net.value], feed_dict={net.observations:[obs]})
        y = explore(tfprob[0], epsilon)
        ys[t] = y.astype(np.float32)
        xs[t] = obs
        vals[t] = v[0]
        obs, reward, done, _ = net.env.step(y if net.vector_action else y[0])
        drs[t] = reward 
        episode_reward += np.mean(reward) * multiplier
        multiplier *= FLAGS.gamma
        if t == FLAGS.batch_size - 1 and not done:
          vals[-1] = sess.run(net.value, feed_dict={net.observations: [obs]})[0]
          train(sess, net, None, xs, ys, vals, drs)
          sess.run(net.update_local)
        if done: break
      if t != FLAGS.batch_size - 1 or done:
        vals[t+1] = 0 if done else sess.run(net.value, feed_dict={
          net.observations: [obs]})[0,0]
        s = train(sess, net, net.summary, xs[:t+1], ys[:t+1], vals[:t+2], drs[:t+2])
        writer.add_summary(s, episode_num)
        epsilon -= (epsilon - end_epsilon) / (FLAGS.total_episodes - e)

      episode_rewards[e % FLAGS.summary_rate] = episode_reward
      if e % FLAGS.summary_rate == FLAGS.summary_rate - 1:
        reward_mean = np.mean(episode_rewards)
        print("Reward mean", reward_mean)
        s = sess.run(net.avg_summary, feed_dict={net.avg_r:reward_mean})
        writer.add_summary(s, episode_num)

      if ((e % FLAGS.save_rate) == 0 or e == FLAGS.total_episodes - 1) \
          and threading.current_thread() == threading.main_thread():
        print("Saving")
        save(global_step=episode_num)
    except KeyboardInterrupt:
      if threading.current_thread() == threading.main_thread():
        print("Saving before exit")
        save(global_step=episode_num)
        raise
