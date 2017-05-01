import numpy as np
from gym_traffic.algorithms.util import *
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

def epoch(sess, env, cmd):
  rnn = None
  obs = env.reset()
  for t in range(FLAGS.episode_len):
    fd = {'observations:0': [obs]}
    if rnn is not None: fd['state_in:0'] = rnn
    y,rnn = sess.run([cmd,"state_out:0"], feed_dict=fd)
    obs, reward, done, _ = env.step(y[0])
    yield (t,obs,y[0].astype(np.float32),reward)
    if done: break

def train(sess, dbg, writer, save, env):
  xs = np.empty((FLAGS.episode_len, *env.observation_space.shape), dtype=np.float32)
  rs = np.empty((FLAGS.episode_len, env.action_space.size), dtype=np.float32)
  ys = np.empty((FLAGS.episode_len, env.action_space.size), dtype=np.float32)
  episode_num = sess.run("episode_num:0")
  try:
    while FLAGS.total_episodes is None or episode_num < FLAGS.total_episodes:
      episode_num = sess.run("episode_num:0")
      sess.run("incr_episode")
      for (t,s,a,r) in epoch(sess, env, "explore:0"):
        ys[t] = a
        xs[t] = s
        rs[t] = r
      epr = rs[:t+1]
      discount(epr, np.float32(FLAGS.gamma), FLAGS.use_avg)
      if not FLAGS.use_avg:
        epr -= np.mean(epr)
        epr /= (np.std(epr) + EPS)
      fd = {"observations:0": xs[:t+1], "actions:0": ys[:t+1], "rewards:0": epr}
      if episode_num % FLAGS.summary_rate == 0:
        _, s = sess.run(["train", "summaries:0"], fd)
        writer.add_summary(s, episode_num)
      else: sess.run("train", fd)
      if episode_num % FLAGS.batch_size == FLAGS.batch_size - 1:
        sess.run("apply_grads")
        sess.run("reset")
      if episode_num % FLAGS.validate_rate == 0:
        rew = validate(sess, env)
        print("Reward", rew)
        s = sess.run("avg_r_summary:0", feed_dict={"avg_r:0":rew})
        writer.add_summary(s, episode_num)
      if episode_num % FLAGS.save_rate == 0:
        save(global_step=episode_num)
  finally:
    save(global_step=episode_num)

def model(env):
  episode_num = tf.Variable(0,dtype=tf.int32,name='episode_num',trainable=False)
  tf.assign_add(episode_num, 1, name="incr_episode")
  eps = exploration_param()
  observations = tf.placeholder(tf.float32, [None,*env.observation_space.shape], name="observations")
  reshape0 = tf.reshape(observations, [-1, env.observation_space.size]) 
  # pre_gru = tf.layers.dense(reshape0, 60, tf.nn.relu, name="pre_gru_layer")
  pre_gru = reshape0
  gru = rnn.GRUCell(80)
  state_in = tf.identity(gru.zero_state(1, tf.float32), name="state_in")
  rnn_out, state_out = tf.nn.dynamic_rnn(gru,
    tf.expand_dims(pre_gru, 0), initial_state=state_in, dtype=tf.float32)
  tf.identity(state_out, name="state_out")
  mid = tf.squeeze(rnn_out, 0, name="mid")
  # h0 = tf.layers.dense(mid, 40, tf.nn.relu, name="hidden_layer")
  h0 = mid
  score = tf.layers.dense(h0, env.action_space.size, name="score_layer")
  sigmoid_decision(score, eps)
  actions = tf.placeholder(tf.float32, [None, env.action_space.size], name="actions")
  rewards = tf.placeholder(tf.float32, [None, env.action_space.size], name="rewards")
  loss = tf.reduce_mean(tf.reduce_sum(rewards *
    tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=actions, name="cross_entropy"),
    axis=1), name="policy_loss")
  tf.summary.scalar("loss", loss)
  opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) 
  grads = [(tf.Variable(tf.zeros(v.get_shape()), trainable=False), g, v)
        for (g,v) in opt.compute_gradients(loss)]
  tf.group(*[ng.assign(tf.zeros_like(ng)) for (ng, _,_) in grads], name="reset")
  tf.group(*[ng.assign_add(g) for (ng, g, _) in grads], name="train")
  opt.apply_gradients([(ng, v) for (ng, _, v) in grads], name="apply_grads")
  tf.identity(tf.summary.merge_all(), name="summaries")

def validate(sess, env):
  return episode_reward(epoch(sess, env, "greedy:0"))

def run(env_f):
  handle_modes(env_f, model, validate, train)
