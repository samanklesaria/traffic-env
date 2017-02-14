import numpy as np
from gym_traffic.algorithms.util import *
import tensorflow as tf
import tensorflow.contrib.layers as tl
import os.path

EPS=1e-8

class PolGradNet:
  def __init__(self, env):
    add_rl_vars(self, env)
    reshaped = tf.reshape(self.observations, [-1, env.observation_space.size])

    self.score = tl.fully_connected(reshaped, num_outputs=env.action_space.size,
        activation_fn=None)
    tf.summary.histogram("scores", self.score)
    self.probs = tf.nn.sigmoid(self.score)
    self.avg_r = tf.placeholder(tf.float32, name="avg_r")
    tf.summary.scalar("avg_r_summary", self.avg_r)
    self.input_y = tf.placeholder(tf.float32, [None, env.action_space.size], name="input_y")
    self.advantages = tf.placeholder(tf.float32, [None, env.reward_size], name="reward_signal")
    assert env.reward_size == env.action_space.size or env.reward_size == 1
    self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(self.score, self.input_y)
    tf.summary.histogram("cross entropy", self.cross_entropy)
    self.loss = tf.reduce_mean(self.advantages * self.cross_entropy)
    tf.summary.scalar("loss", self.loss)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) 
    grads = [(tf.Variable(tf.zeros(v.get_shape()), trainable=False), g, v)
          for (g,v) in opt.compute_gradients(self.loss)]
    entropy = -tf.reduce_mean(self.probs * tf.log(self.probs + EPS))
    tf.summary.scalar("entropy", entropy)
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
      load_from_checkpoint(sess)
      sess.run(net.reset)
      while True:
        print("Validation reward", validate(net, env, sess))
  if tf.gfile.Exists(FLAGS.logdir):
    tf.gfile.DeleteRecursively(FLAGS.logdir)
  tf.gfile.MakeDirs(FLAGS.logdir)
  writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, "polgrad"))
  xs = np.empty((FLAGS.episode_len, *env.observation_space.shape), dtype=np.float32)
  drs = np.empty((FLAGS.episode_len, env.reward_size), dtype=np.float32)
  ys = np.empty((FLAGS.episode_len, env.action_space.size), dtype=np.float32)
  reward_sum = 0
  epsilon = 1
  end_epsilon = 0.1

  explore = globals()[FLAGS.exploration]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    net.save = load_from_checkpoint(sess)
    sess.run(net.reset)
    episode_number = 0
    try:
      # env.unwrapped.reset_entrypoints()
      for e in range(FLAGS.total_episodes):
        episode_number = sess.run(net.episode_num)
        sess.run(net.increment_episode)
        observation = env.reset()
        multiplier = 1.0
        for t in range(FLAGS.episode_len):
          tfprob, = sess.run(net.probs,
              feed_dict={net.observations: [observation]})
          flaty = explore(tfprob, epsilon)
          ys[t] = flaty.astype(np.float32)
          y = np.reshape(flaty, env.action_space.shape)
          xs[t] = observation
          observation, reward, done, _ = env.step(y)
          reward_sum += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
          multiplier *= FLAGS.gamma
          drs[t] = reward
          if done: break
        epr = drs[:t+1]
        
        # SOMETHING IS profoundly broken

        # old_disc = discount(epr, FLAGS.gamma)
        epr = my_discount(epr, FLAGS.gamma)
        # assert (np.abs(epr - old_disc) < 0.0001).all()
        epr -= np.mean(epr)
        epr /= (np.std(epr) + EPS)
        fd = {net.observations: xs[:t+1], net.input_y: ys[:t+1], net.advantages: epr}
        if episode_number % FLAGS.batch_size == FLAGS.batch_size - 1:
          fd[net.avg_r] = reward_sum / FLAGS.batch_size
          _, s = sess.run([net.train, net.summary], feed_dict=fd)
          writer.add_summary(s, episode_number)
          sess.run(net.apply_grads)
          sess.run(net.reset)
          print(episode_number, "average reward", fd[net.avg_r])
          if (episode_number % FLAGS.save_rate) == 0 or e == FLAGS.total_episodes - 1:
            print("Saving")
            net.save(global_step=episode_number)
          reward_sum = 0
          # env.unwrapped.reset_entrypoints()
        else: sess.run(net.train, feed_dict=fd)
        epsilon -= (epsilon - end_epsilon) / (FLAGS.total_episodes - e)
    except KeyboardInterrupt:
      print("Saving before exit")
      net.save(global_step=episode_number)
      raise

def validate(net, env, sess):
  reward_sum = 0
  obs = env.reset()
  multiplier = 1.0
  for _ in range(FLAGS.episode_len):
    dist, = sess.run(net.probs,feed_dict={net.observations: [obs]})
    if FLAGS.render: print("Action", dist)
    y = np.reshape(proportional(dist, None), env.action_space.shape).astype(np.int8)
    obs, reward, done, _ = env.step(y)
    if FLAGS.render: print("Reward", reward)
    reward_sum += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1) 
    multiplier *= FLAGS.gamma
    if done: break
  return reward_sum
