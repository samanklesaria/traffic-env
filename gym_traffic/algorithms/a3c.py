import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
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

# Makes an image from 5-channel observations
# We should do this manually.
def obs_image(obs, obs_shape):
  history, snapshots, channels, width, height, = obs_shape
  channel_first = tf.transpose(obs, perm=[3,0,1,2,4,5])
  waiting = tf.gather(channel_first, [4,4,4,4])
  waiting_mag = tf.abs(waiting)
  normalized_mag = waiting_mag * (255 /
    (0.01 * FLAGS.light_secs / FLAGS.rate))
  reds = tf.cast(waiting > 0, tf.float32) * normalized_mag
  greens = tf.cast(waiting < 0, tf.float32) * normalized_mag
  passed = channel_first[:4]
  blues = 255 - passed * 255 
  colored = tf.stack([reds, greens, blues])
  reshaped = tf.reshape(colored, [3, 2, 2, -1, history, snapshots, width, height])
  transposed = tf.transpose(reshaped, perm=[3,5,7,2,4,6,1,0])
  squished = tf.reshape(transposed, (-1, snapshots, height *2, history, width*2, 3))
  padded = tf.pad(squished, [[0,0],[1,0],[1,0],[1,0],[1,0],[0,0]])
  pic = tf.reshape(padded, (-1, (snapshots+1) * (1+(height*2)), (history+1) * ((2*width)+1), 3))
  return tf.cast(255 - pic[:,height*2+1:,width*2+1:], tf.uint8), blues

# Problem with showing this in the visualization is I have to worry about scaling. Still, doable. 
# If we're doing that, we probably don't want this representation. We want to put it into
# the visualization. Which we can do in numpy. 
# What if, when visualizing, the internal rate got more detailed (0.1). The strobe compensated. 
# The obs was also shown. Then no icky tensorflow hacks.

# Contains the network and environment for a single thread
class A3CNet:
  def __init__(self, env_f):
    self.name = tf.get_variable_scope().name
    self.env = env_f()
    add_rl_vars(self, self.env)
    hist = self.env.observation_space.shape[:-3]
    channels = self.env.observation_space.shape[-3]
    all_channels = np.prod(hist) * channels
    dims = self.env.observation_space.shape[-2:]
    flat_obs = tf.reshape(self.observations, [-1, all_channels, *dims])
    nhwc = tf.transpose(flat_obs, perm=[0,3,2,1])
    local = tl.conv2d(nhwc, 80, [1,1])
    mid = tl.conv2d(local, 80, [1,1])
    mid_size = int(np.prod(dims)*80)
    reshaped = tf.reshape(mid, [-1, mid_size])
    collected = tl.fully_connected(reshaped, 150)
    resid_a = tl.fully_connected(collected, 150)
    resid_b = tl.fully_connected(resid_a, 150, activation_fn=None)
    hidden = tf.nn.relu(collected + resid_b)
    self.score = tl.fully_connected(hidden, self.env.action_space.size, activation_fn=None)
    self.probs = tf.nn.sigmoid(self.score)
    tf.summary.histogram("probs", self.probs)
    self.value = tl.fully_connected(hidden, num_outputs=self.env.reward_size, activation_fn=None)
    obs_sum, self.blues = obs_image(self.observations, self.env.observation_space.shape)
    self.obs_image = tf.summary.image("obs_image", obs_sum, 
        max_outputs=20, collections=[])

    # self.prob_grads = tf.map_fn(lambda p: tf.gradients(p, [self.observations])[0][0],
    #     tf.reshape(self.probs[0], [-1]))

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
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("entropy", entropy)
    tf.summary.scalar("value_loss", value_loss)
    tf.summary.scalar("policy_loss", policy_loss)
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    gradients = tf.gradients(loss,local_vars)
    self.grads,_ = tf.clip_by_global_norm(gradients,100.0)
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

def validate(net, env, sess, writer):
  old_entry = FLAGS.entry
  if not FLAGS.validate: FLAGS.entry = "all"
  env.unwrapped.reset_entrypoints()
  net.env.unwrapped.seed_generator(None)
  reward_sum = 0
  obs = env.reset()
  multiplier = 1.0
  for i in range(FLAGS.episode_len):
    if FLAGS.obs_pic:
      s,b = sess.run([net.obs_image, net.blues], feed_dict={net.observations: [obs]})
      # print("Blues", b)
      writer.add_summary(s, i)
      if FLAGS.render: writer.flush()
    dist, = sess.run(net.probs, feed_dict={net.observations: [obs]})
    if FLAGS.render: print("Action", dist)
    y = env.action_space.to_action(proportional(dist, None))
    obs, reward, done, _ = env.step(y)
    reward_sum += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
    multiplier *= FLAGS.gamma
    if done: break
  FLAGS.entry = old_entry
  env.unwrapped.reset_entrypoints()
  return reward_sum

def run(env_f):
  with tf.device("/cpu:0"):
    with tf.variable_scope('global'): master = A3CNet(env_f)
    if not FLAGS.restore: 
      if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)
      tf.gfile.MakeDirs(FLAGS.logdir)
    else:
      if tf.gfile.Exists("validation_summary"):
        tf.gfile.DeleteRecursively("validation_summary")
      tf.gfile.MakeDirs("validation_summary")
    if not FLAGS.validate:
      opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      workers = [make_worker('w'+str(t), env_f) for t in range(FLAGS.threads)]
      with tf.variable_scope('application'):
        for w in workers: w.make_apply_ops(opt)
      if not FLAGS.restore:
        gw = tf.summary.FileWriter(os.path.join(FLAGS.logdir, "graph"),
            tf.get_default_graph())
        gw.close()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      master.save = load_from_checkpoint(sess,
          tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'))
      if FLAGS.validate:
        writer = tf.summary.FileWriter(os.path.join("validation_summary", "results"))
        def rewards():
          while True:
            yield validate(master, master.env, sess, writer)
        print_running_stats(rewards())
      else:
        threads = [threading.Thread(target=work, args=[w, sess, None]) for w in workers[1:]]
        for t in threads: t.start()
        work(workers[0], sess, master.save)

def train(sess, net, summary, xs, ys, vals, drs):
  drs[-1] = vals[-1]
  advantages = drs[:-1] + FLAGS.gamma * vals[1:] - vals[:-1]
  drs = discount(drs, FLAGS.lam * FLAGS.gamma)
  advantages = discount(advantages, FLAGS.gamma)
  fd = {net.observations: xs, net.input_y: ys,
    net.advantages: advantages, net.target_v: drs[:-1]}
  return sess.run([net.apply_grads,summary], feed_dict=fd)[1]

def work(net, sess, save):
  writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, net.name))
  ys = np.empty((FLAGS.batch_size, net.env.action_space.size), dtype=np.float32)
  vals = np.empty((FLAGS.batch_size + 1, net.env.reward_size), dtype=np.float32)
  xs = np.empty((FLAGS.batch_size, *net.env.observation_space.shape), dtype=np.float32)
  drs = np.empty((FLAGS.batch_size + 1, net.env.reward_size), dtype=np.float32)
  episode_rewards = np.zeros(FLAGS.summary_rate, dtype=np.float32)
  epsilon = 1
  end_epsilon = np.random.uniform(0.05, 0.2)
  print("Started worker", net.name, "with target epsilon", end_epsilon)
  explore = globals()[FLAGS.exploration]

  for e in range(FLAGS.total_episodes):
    episode_num = sess.run(net.episode_num)
    sess.run(net.increment_episode)
    seed = np.random.randint(1000)
    try:
      sess.run(net.update_local)
      net.env.unwrapped.seed_generator(seed)
      net.env.unwrapped.reset_entrypoints()
      obs = net.env.reset()
      s = None
      for mt in range(FLAGS.episode_len):
        t = mt % FLAGS.batch_size
        tfprob,v = sess.run([net.probs,net.value], feed_dict={net.observations:[obs]})
        flaty = explore(tfprob[0], epsilon)
        ys[t] = flaty.astype(np.float32)
        y = net.env.action_space.to_action(flaty)
        xs[t] = obs
        vals[t] = v[0]
        obs, reward, done, _ = net.env.step(y)
        drs[t] = reward / 100.0
        if t == FLAGS.batch_size - 1 and not done:
          vals[-1] = sess.run(net.value, feed_dict={net.observations: [obs]})[0]
          s = train(sess, net, net.summary, xs, ys, vals, drs)
          sess.run(net.update_local)
        if done: break
      if t != FLAGS.batch_size - 1 or done:
        vals[t+1] = 0 if done else sess.run(net.value, feed_dict={
          net.observations: [obs]})[0,0]
        s = train(sess, net, net.summary, xs[:t+1], ys[:t+1], vals[:t+2], drs[:t+2])
      writer.add_summary(s, episode_num)
      epsilon -= (epsilon - end_epsilon) / (FLAGS.total_episodes - e)

      if e % FLAGS.summary_rate == FLAGS.summary_rate - 1:
        reward_mean = validate(net, net.env, sess, writer)
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
