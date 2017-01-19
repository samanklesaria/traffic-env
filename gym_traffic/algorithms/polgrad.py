import numpy as np
from gym_traffic.algorithms import *
import tensorflow as tf
import tensorflow.contrib.layers as tl
import os.path

flags.DEFINE_integer('batch_size', 5, 'Update params every how many episodes')

class PolGradNet(TFAgent):
  def __init__(self, env):
    super().__init__(env)
    hidden = tl.fully_connected(self.flat_obs, num_outputs=10)
    self.score = tl.fully_connected(hidden, num_outputs=self.num_actions, activation_fn=None)
    self.probs = tf.nn.sigmoid(self.score)
    self.avg_r = tf.placeholder(tf.float32, name="avg_r")
    tf.summary.scalar("avg_r_summary", self.avg_r)
    self.input_y = tf.placeholder(tf.float32, [None, self.num_actions], name="input_y")
    self.advantages = tf.placeholder(tf.float32, [None, self.num_actions], name="reward_signal")
    loss = tf.reduce_mean(self.advantages *
        tf.nn.sigmoid_cross_entropy_with_logits(self.score, self.input_y))
    tf.summary.scalar("loss", loss)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) 
    grads = [(tf.Variable(tf.zeros(v.get_shape()), trainable=False), g, v)
          for (g,v) in opt.compute_gradients(loss)]
    self.reset = [ng.assign(tf.zeros_like(ng)) for (ng, _,_) in grads]
    self.train = [ng.assign_add(g) for (ng, g, _) in grads]
    self.apply_grads = opt.apply_gradients([(ng, v) for (ng, _, v) in grads])
    self.summary = tf.summary.merge_all()

EPS = 1e-6

def run(env_f):
  env = env_f()
  net = PolGradNet(env)
  if FLAGS.validate:
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      net.load_from_checkpoint(sess)
      sess.run(net.reset)
      while True:
        print(validate(net, env, sess))
  if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
  tf.gfile.MakeDirs(FLAGS.logdir)
  writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, "polgrad"))
  xs = np.empty((FLAGS.episode_len, *env.observation_space.shape), dtype=np.float32)
  drs = np.empty((FLAGS.episode_len, net.num_actions), dtype=np.float32)
  ys = np.empty((FLAGS.episode_len, net.num_actions), dtype=np.float32)
  reward_sum = 0

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    net.load_from_checkpoint(sess)
    sess.run(net.reset)
    for episode_number in range(FLAGS.total_episodes):
      observation = env.reset()
      for t in range(FLAGS.episode_len):
        tfprob, = sess.run(net.probs,feed_dict={net.observations: [observation]})
        y = (np.random.uniform(size=tfprob.shape) < tfprob).astype(np.int8)
        ys[t] = y.astype(np.float32)
        xs[t] = observation
        observation, reward, done, _ = env.step(y if net.vector_action else y[0])
        reward_sum += np.sum(reward)
        drs[t] = reward
        if done: break
      epr = drs[:t+1]
      discount(epr, FLAGS.gamma)
      epr -= np.mean(epr)
      epr /= (np.std(epr) + EPS)
      fd = {net.observations: xs[:t+1], net.input_y: ys[:t+1], net.advantages: epr}
      if episode_number % FLAGS.batch_size == 0:
        fd[net.avg_r] = reward_sum / FLAGS.batch_size
        _, s = sess.run([net.train, net.summary], feed_dict=fd)
        writer.add_summary(s, episode_number)
        sess.run(net.apply_grads)
        sess.run(net.reset)
        print(episode_number, "average reward", fd[net.avg_r])
        if ((episode_number // FLAGS.batch_size) % FLAGS.save_rate) == 0:
          print("Saving")
          net.saver.save(sess, net.checkpoint_file)
        reward_sum = 0
      else: sess.run(net.train, feed_dict=fd)

def validate(net, env, sess):
  reward_sum = 0
  obs = env.reset()
  for _ in range(FLAGS.episode_len):
    dist = sess.run(net.probs,feed_dict={net.observations: [obs]})
    if FLAGS.render: print("Action", dist)
    y, = np.round(dist).astype(np.int8)
    obs, reward, done, _ = env.step(y if net.vector_action else y[0])
    if FLAGS.render: print("Reward", reward)
    reward_sum += reward
    if done: break
  return reward_sum
