import numpy as np
from gym_traffic.algorithms import *
import tensorflow as tf
import tensorflow.contrib.layers as tl
import os.path

EPS = 1e-6

class PolGradNet(TFAgent):
  def __init__(self, env):
    super().__init__(env)
    channels = np.prod(env.observation_space.shape[:-2])
    dims = env.observation_space.shape[-2:]
    flat_obs = tf.reshape(self.observations, [-1, channels, *dims])
    nhwc = tf.transpose(flat_obs, perm=[0,2,3,1])
    local = tl.convolution(nhwc, 30, [1,1])
    reshaped = tf.reshape(local, [-1, 30*self.num_actions])
    resid_a = tl.fully_connected(reshaped, self.num_actions)
    resid_b = tl.fully_connected(resid_a, 30*self.num_actions, activation_fn=None)
    mid = tf.nn.relu(resid_b + reshaped)
    self.score = tl.fully_connected(mid, self.num_actions, activation_fn=None)
    self.probs = tf.nn.sigmoid(self.score)
    entropy = -tf.reduce_mean(tf.reduce_sum(self.probs * tf.log(self.probs + EPS), axis=1))
    tf.summary.scalar("entropy", entropy)
    self.avg_r = tf.placeholder(tf.float32, name="avg_r")
    tf.summary.scalar("avg_r_summary", self.avg_r)
    self.input_y = tf.placeholder(tf.float32, [None, self.num_actions], name="input_y")
    self.advantages = tf.placeholder(tf.float32, [None, self.num_actions], name="reward_signal")
    loss = tf.reduce_mean(self.advantages *
        tf.nn.sigmoid_cross_entropy_with_logits(self.score, self.input_y)) - 0.001 * entropy
    tf.summary.scalar("loss", loss)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) 
    grads = [(tf.Variable(tf.zeros(v.get_shape()), trainable=False), g, v)
          for (g,v) in opt.compute_gradients(loss)]
    self.reset = [ng.assign(tf.zeros_like(ng)) for (ng, _,_) in grads]
    self.train = [ng.assign_add(g) for (ng, g, _) in grads]
    self.apply_grads = opt.apply_gradients([(ng, v) for (ng, _, v) in grads])
    self.summary = tf.summary.merge_all()

def run(env_f):
  env = env_f()
  net = PolGradNet(env)
  if FLAGS.validate:
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      net.load_from_checkpoint(sess)
      sess.run(net.reset)
      while True:
        print("Validation reward", validate(net, env, sess))
  if not FLAGS.restore:
    if tf.gfile.Exists(FLAGS.logdir):
      tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.gfile.MakeDirs(FLAGS.logdir)
  writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, "polgrad"),
      tf.get_default_graph())
  xs = np.empty((FLAGS.episode_len, *env.observation_space.shape), dtype=np.float32)
  drs = np.empty((FLAGS.episode_len, net.num_actions), dtype=np.float32)
  ys = np.empty((FLAGS.episode_len, net.num_actions), dtype=np.float32)
  reward_sum = 0
  epsilon = 1
  end_epsilon = 0.1

  explore = globals()[FLAGS.exploration]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    net.load_from_checkpoint(sess)
    sess.run(net.reset)
    episode_number = 0
    try:
      for e in range(FLAGS.total_episodes):
        episode_number = sess.run(net.episode_num)
        sess.run(net.increment_episode)
        observation = env.reset()
        multiplier = 1.0
        for t in range(FLAGS.episode_len):
          tfprob, = sess.run(net.probs,feed_dict={net.observations: [observation]})
          y = explore(tfprob, epsilon)
          ys[t] = y.astype(np.float32)
          xs[t] = observation
          observation, reward, done, _ = env.step(y if net.vector_action else y[0])
          reward_sum += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
          multiplier *= FLAGS.gamma
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
          if (episode_number % FLAGS.save_rate) == 0 or e == FLAGS.total_episodes - 1:
            print("Saving")
            net.saver.save(sess, net.checkpoint_file, global_step=episode_number)
          reward_sum = 0
        else: sess.run(net.train, feed_dict=fd)
        epsilon -= (epsilon - end_epsilon) / (FLAGS.total_episodes - e)
    except KeyboardInterrupt:
      print("Saving before exit")
      net.saver.save(sess, net.checkpoint_file, global_step=episode_number)
      raise

def validate(net, env, sess):
  reward_sum = 0
  obs = env.reset()
  multiplier = 1.0
  for _ in range(FLAGS.episode_len):
    dist = sess.run(net.probs,feed_dict={net.observations: [obs]})
    if FLAGS.render: print("Action", dist)
    y, = np.round(dist).astype(np.int8)
    obs, reward, done, _ = env.step(y if net.vector_action else y[0])
    if FLAGS.render: print("Reward", reward)
    reward_sum += np.mean(reward) * multiplier
    multiplier *= FLAGS.gamma
    if done: break
  return reward_sum
