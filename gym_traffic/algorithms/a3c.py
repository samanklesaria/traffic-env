import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from gym_traffic.algorithms.util import *
import threading
from functools import partial

def build_net(env, opt, eps, name):
  with tf.variable_scope(name):
    episode_num = tf.Variable(0,dtype=tf.int32,name='episode_num',trainable=False)
    tf.assign_add(episode_num, 1, name="incr_episode")
    obs_shape = env.observation_space.shape
    observations = tf.placeholder(tf.float32, [None,*obs_shape], name="observations")
    pre_gru = tf.layers.dense(observations, 160, tf.nn.relu)
    gru = rnn.GRUCell(160)
    state_in = tf.identity(gru.zero_state(1, tf.float32), name="state_in")
    rnn_out, state_out = tf.nn.dynamic_rnn(gru,
        tf.expand_dims(pre_gru, 0), initial_state=state_in, dtype=tf.float32)
    tf.identity(state_out, name="state_out")
    mid = tf.squeeze(rnn_out, 0, name="mid")
    h0 = tf.layers.dense(mid, 160, tf.nn.relu, name="h0")
    scores = tf.layers.dense(h0, env.action_space.size, name="score_layer")
    sigmoid_decision(scores, eps)
    value = tf.identity(tf.layers.dense(h0, env.reward_size, name="value_layer"), name="value")
    if opt:
      tf.group(*[dst.assign(src) for src, dst in zip(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'),
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name))], name="update_local")
      target_v = tf.placeholder(tf.float32, [None, env.reward_size], name="target_v")
      actions = tf.placeholder(tf.float32, [None, env.action_space.size], name="actions")
      advantages = tf.placeholder(tf.float32, [None, env.reward_size], name="advantages")
      policy_loss = tf.reduce_mean(tf.reduce_sum(advantages * tf.nn.sigmoid_cross_entropy_with_logits(
          logits=scores, labels=actions), axis=1))
      tf.summary.scalar("policy_loss", policy_loss)
      value_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(target_v - value), axis=1), axis=0)
      tf.summary.scalar("value_loss", value_loss)
      loss = 0.5 * value_loss + policy_loss - ref(name+"/entropy:0") * 0.001
      tf.summary.scalar("loss", loss)
      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
      grads,_ = tf.clip_by_global_norm(tf.gradients(loss,local_vars),40.0)
      if FLAGS.grad_summary:
        prob_grads = [tf.gradients(probs[0][i],
            [observations, state_in]) for i in range(env.action_space.size)]
        for (i,g) in enumerate(prob_grads):
          tf.summary.histogram("obs_grad"+str(i), g[0])
          tf.summary.histogram("state_grad"+str(i), g[1])
      global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
      apply_grads = opt.apply_gradients(zip(grads, global_vars), name="apply_grads")
      tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, name), name="desc")

# Generator to simulate an epoch
def epoch(scope, sess, env, cmd, extra_ops=[]):
  rnn = None
  obs = env.reset()
  ops = [scope + "/" + s for s in [cmd, "state_out:0", *extra_ops]]
  for mt in range(FLAGS.episode_len):
    fd = {scope + "/observations:0": [obs]}
    if rnn is not None: fd[scope + "/state_in:0"] = rnn
    y,rnn,*others = sess.run(ops, feed_dict=fd)
    new_obs, reward, done, _ = env.step(y[0])
    yield (mt,obs,y[0].astype(np.float32),reward,new_obs,done,*others)
    obs = new_obs
    if done: break

# Run A3C training method
def train_model(env_f, sess, dbg, writer, save, master_env):
  ev = threading.Event()
  coord = tf.train.Coordinator()
  envs = [env_f() for _ in range(FLAGS.threads)]
  threads = [threading.Thread(target=work, args=["w"+str(i),e,sess,coord,ev])
      for (i,e) in enumerate(envs)]
  for t in threads: t.start()
  episode_num = sess.run("global/episode_num:0")
  try:
    while not coord.should_stop():
      episode_num = sess.run("global/episode_num:0")
      sess.run("global/incr_episode")
      ev.wait()
      if (episode_num % FLAGS.summary_rate) == 0:
        rew = validate(sess, master_env)
        print("Reward", rew)
        s = sess.run("avg_r_summary:0", feed_dict={"avg_r:0":rew})
        writer.add_summary(s, episode_num)
      if (episode_num % FLAGS.save_rate) == 0:
        save(global_step=episode_num)
      ev.clear()
  finally:
    print("Waiting for threads to stop")
    coord.request_stop()
    coord.join(threads)
    print("Saving")
    save(global_step=episode_num)

# Running training step on previous epoch, using generalized advantage
def train(sess, scope, xs, ys, vals, drs):
  drs[-1] = vals[-1]
  deltas = drs[:-1] + FLAGS.gamma * vals[1:] - vals[:-1]
  drs = discount(drs, FLAGS.gamma)
  advantages = discount(deltas, FLAGS.lam * FLAGS.gamma)
  fd = {scope+'/observations:0': xs, scope+'/actions:0': ys,
    scope+'/advantages:0': advantages, scope+'/target_v:0': drs[:-1]}
  return sess.run([scope+'/apply_grads',scope+'/desc/desc:0'], feed_dict=fd)[1]

# Run an A3C worker thread
def work(scope, env, sess, coord, ev):
  writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, scope))
  ys = np.empty((FLAGS.batch_size, env.action_space.size), dtype=np.float32)
  vals = np.empty((FLAGS.batch_size + 1, env.reward_size), dtype=np.float32)
  xs = np.empty((FLAGS.batch_size, *env.observation_space.shape), dtype=np.float32)
  drs = np.empty((FLAGS.batch_size + 1, env.reward_size), dtype=np.float32)
  print("Started worker", scope)
  while not coord.should_stop():
    episode_num = sess.run(scope + "/episode_num:0")
    sess.run(scope + "/incr_episode")
    sess.run(scope + "/update_local")
    for (mt,obs,y,reward,new_obs,done,v) in epoch(scope,sess,env,"explore:0",["value:0"]):
      t = mt % FLAGS.batch_size
      ys[t] = y
      xs[t] = obs
      vals[t] = v[0]
      drs[t] = reward / 100.0
      if t == FLAGS.batch_size - 1 and not done:
        vals[-1] = sess.run(scope + "/value:0", feed_dict={scope+"/observations:0": [new_obs]})[0]
        s = train(sess, scope, xs, ys, vals, drs)
        sess.run(scope+'/update_local')
      if done: break
    if t != FLAGS.batch_size - 1 or done:
      vals[t+1] = 0 if done else sess.run(scope + "/value:0", feed_dict={
        scope+"/observations:0": [obs]})[0,0]
      s = train(sess, scope, xs[:t+1], ys[:t+1], vals[:t+2], drs[:t+2])
    writer.add_summary(s, episode_num)
    ev.set()

def model(env):
  opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  eps = exploration_param()
  build_net(env, None, eps, "global") 
  for i in range(FLAGS.threads): build_net(env, opt, eps, "w"+str(i)) 

def validate(sess, env):
  return episode_reward(epoch("global", sess, env, "greedy:0"))

def run(env_f):
  handle_modes(env_f, model, validate, partial(train_model, env_f))
