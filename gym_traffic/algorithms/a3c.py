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

def build_net(env, opt, add_train_ops, name):
  with tf.variable_scope(name):
    obs_shape = env.observation_space.shape
    episode_num = tf.Variable(0,dtype=tf.int32,name='episode_num',trainable=False)
    tf.stop_gradient(episode_num.assign_add(1), name="increment_episode")
    observations = tf.placeholder(tf.float32, [None,*obs_shape], name="observations")
    gru = rnn.GRUCell(120)
    state_in = tf.identity(gru.zero_state(1, tf.float32), name="state_in")
    rnn_out, state_out = tf.nn.dynamic_rnn(gru,
        tf.expand_dims(observations, 0), initial_state=state_in, dtype=np.float32)
    tf.identity(rnn_out, name="rnn_out")
    tf.identity(state_out, name="state_out")
    mid = tf.squeeze(rnn_out, 0)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, mid)
    hidden = tl.fully_connected(mid, 100, outputs_collections=tf.GraphKeys.ACTIVATIONS)
    score = tl.fully_connected(hidden, env.action_space.size, activation_fn=None)
    probs = tf.nn.sigmoid(score, name="probs")
    value = tf.identity(tl.fully_connected(hidden, num_outputs=env.reward_size, activation_fn=None),
        name="value")
    tf.summary.scalar("value_sum", tf.reduce_sum(value))
    entropy = tf.negative(tf.reduce_mean(probs * tf.log(probs + EPS)))
    tf.summary.scalar("entropy", entropy)

    if add_train_ops:
      tf.group(*[dst.assign(src) for src, dst in zip(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'),
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name))], name="update_local")
      target_v = tf.placeholder(tf.float32, [None, env.reward_size], name="target_v")
      actions = tf.placeholder(tf.float32, [None, env.action_space.size], name="actions")
      advantages = tf.placeholder(tf.float32, [None, env.reward_size], name="advantages")
      policy_loss = tf.reduce_sum(advantages * tf.nn.sigmoid_cross_entropy_with_logits(
          logits=score, labels=actions))
      tf.summary.scalar("policy_loss", policy_loss)
      value_loss = 0.5 * tf.reduce_sum(tf.square(target_v - value))
      tf.summary.scalar("value_loss", value_loss)
      loss = 0.5 * value_loss + policy_loss - entropy * 0.001
      tf.summary.scalar("loss", loss)
      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
      grads,_ = tf.clip_by_global_norm(tf.gradients(loss,local_vars),40.0)
      if FLAGS.grad_summary:
        prob_grads = [tf.gradients(probs[0][i],
            [observations, state_in]) for i in range(env.action_space.size)]
        for (i,g) in enumerate(prob_grads):
          tf.summary.histogram("obs_grad"+str(i), g[0])
          tf.summary.histogram("state_grad"+str(i), g[1])
      tl.summarize_activations(name)
      global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
      apply_grads = opt.apply_gradients(zip(grads, global_vars), name="apply_grads")
    else:
      tf.summary.scalar("avg_r_summary", tf.placeholder(tf.float32, name="avg_r"))
    tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, name), name="desc")

# Generator to simulate an epoch
def epoch(scope, sess, env, extra_ops=[]):
  rnn = None
  obs = env.reset()
  ops = [scope + "/" + s for s in ["probs:0", "state_out:0", *extra_ops]]
  for mt in range(FLAGS.episode_len):
    fd = {scope + "/observations:0": [obs]}
    if rnn is not None: fd[scope + "/state_in:0"] = rnn
    dist,rnn,*others = sess.run(ops, feed_dict=fd)
    flaty = proportional(dist[0], None)
    y = env.action_space.to_action(flaty)
    new_obs, reward, done, _ = env.step(y)
    yield mt,obs,rnn,dist[0],flaty.astype(np.float32),reward,new_obs,done,others
    obs = new_obs
    if done: break

def validate(sess, env, writer):
  reward_sum = 0
  multiplier = 1.0
  for (i,obs,rnn,d,_,r,_,_,others) in epoch("global", sess, env):
    if FLAGS.render:
      print("Obs", obs)
      print("Hidden", rnn)
      print("Action", d)
    reward_sum += np.mean(r) * (multiplier if FLAGS.print_discounted else 1)
    multiplier *= FLAGS.gamma
  return reward_sum

# Run A3C training method
def train_model(sess, writer, weight_saver, master_env, envs):
  ev = threading.Event()
  coord = tf.train.Coordinator()
  threads = [threading.Thread(target=work, args=["w"+str(i),e,sess,coord,ev])
      for (i,e) in enumerate(envs)]
  model_file = os.path.join(FLAGS.logdir, 'model.ckpt')
  for t in threads: t.start()
  episode_num = None
  try:
    while not coord.should_stop():
      episode_num = sess.run("global/episode_num:0")
      sess.run("global/increment_episode")
      ev.wait()
      if (episode_num % FLAGS.summary_rate) == 0:
        rew = validate(sess, master_env, writer)
        print("Reward", rew)
        s = sess.run("global/avg_r_summary:0", feed_dict={"global/avg_r:0":rew})
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

def run(env_f, derive_flags):
  master_env = env_f()
  if not FLAGS.restore:
    remkdir(FLAGS.logdir)
    with tf.device("/cpu:0"):
      opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      for (k,v) in vars(FLAGS).items():
        tf.add_to_collection("hparams",tf.constant(v, name=k))
      envs = [env_f() for _ in range(FLAGS.threads)]
      build_net(master_env, opt, False, "global")
      for (i,e) in enumerate(envs): build_net(e, opt, True, 'w'+str(i))
  with tf.Session() as sess:
    if FLAGS.restore:
      latest = tf.train.latest_checkpoint(FLAGS.logdir)
      tf.train.import_meta_graph(latest + '.meta').restore(sess, latest)
      hp = tf.get_collection("hparams")
      with (k,v) in zip(hp, sess.run(hp)):
        print("GOT", k.name, v)
        FLAGS.k = v
      derive_flags()
      envs = [env_f() for _ in range(FLAGS.threads)]
    else: sess.run(tf.global_variables_initializer())
    if FLAGS.mode == "validate":
      summary_writer = tf.summary.FileWriter("val_dir", tf.get_default_graph())
      def rewards():
        while True:
          yield validate(sess, master_env, summary_writer)
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
      if FLAGS.restore: summary_writer = tf.summary.FileWriter(FLAGS.logdir)
      else: summary_writer = tf.summary.FileWriter(FLAGS.logdir, tf.get_default_graph())
      weight_saver = tf.train.Saver(max_to_keep=10)
      train_model(sess, summary_writer, weight_saver, master_env, envs)
    else: print("Unknown mode", FLAGS.mode)

# Running training step on previous epoch, using generalized advantage
def train(sess, scope, xs, ys, vals, drs):
  drs[-1] = vals[-1]
  advantages = drs[:-1] + FLAGS.gamma * vals[1:] - vals[:-1]
  drs = discount(drs, FLAGS.lam * FLAGS.gamma)
  advantages = discount(advantages, FLAGS.gamma)
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
  episode_rewards = np.zeros(FLAGS.summary_rate, dtype=np.float32)
  print("Started worker", scope)
  while not coord.should_stop():
    episode_num = sess.run(scope + "/episode_num:0")
    sess.run(scope + "/increment_episode")
    sess.run(scope + "/update_local")
    for (mt,obs,_,_,y,reward,new_obs,done,v) in epoch(scope,sess,env,["value:0"]):
      t = mt % FLAGS.batch_size
      ys[t] = y
      xs[t] = obs
      vals[t] = v[0][0]
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
