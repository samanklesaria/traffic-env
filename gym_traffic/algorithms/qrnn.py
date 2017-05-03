import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from gym_traffic.algorithms.util import *

# max-predicted q is only useful in comparison to discounted rewards.
# We should record both averages and discounted returns

def build_net(env, temp, n_ep, n_exp, observations, lens):
  reshape0 = tf.reshape(observations, [-1, env.observation_space.size]) 
  pre_gru = tf.reshape(tf.layers.dense(reshape0, 180, tf.nn.relu),
      [n_ep, n_exp, 180])
  gru = rnn.GRUCell(180)
  state_in = tf.identity(gru.zero_state(n_ep, tf.float32), name="state_in")
  rnn_out, state_out = tf.nn.dynamic_rnn(gru,
    pre_gru, lens, state_in, dtype=tf.float32)
  tf.identity(state_out, name="state_out")
  reshaped = tf.reshape(rnn_out, [-1, 180]) 
  a_stream, v_stream = tf.split(reshaped, 2, 1)
  advantage = tf.reshape(tf.layers.dense(a_stream, env.action_space.size * 2),
    (n_ep, n_exp, env.action_space.size, 2))
  value = tf.reshape(tf.layers.dense(v_stream, env.action_space.size * 2),
    (n_ep, n_exp, env.action_space.size, 2))
  qvals = tf.subtract(value + advantage, tf.reduce_mean(advantage, axis=-1,
    keep_dims=True), name="qvals")
  softmax_decision(qvals, temp)

def random_trace(n_exp, p):
  l, rlen = p
  start = tf.random_uniform([], maxval=tf.maximum(1,l-n_exp+1), dtype=tf.int32)
  zeros = tf.zeros([n_exp - rlen], dtype=tf.int32)
  return tf.concat([tf.range(start, start + rlen), zeros], 0)

def experience_replay(env, episode_num, n_ep, n_exp):
  with tf.variable_scope("replay"):
    aa = tf.Variable(tf.zeros((FLAGS.buffer_size, FLAGS.episode_len,
      env.action_space.size), dtype=tf.int32), trainable=False, name="action_replay")
    rr = tf.Variable(tf.zeros((FLAGS.buffer_size, FLAGS.episode_len,
      env.reward_size), dtype=tf.float32), trainable=False, name="reward_replay")
    ss = tf.Variable(tf.zeros((FLAGS.buffer_size, FLAGS.episode_len + 1,
      *env.observation_space.shape), dtype=tf.float32), trainable=False, name="obs_replay")
    lens = tf.Variable(tf.zeros([FLAGS.buffer_size],
      dtype=tf.int32), trainable=False, name="len_replay")
    nd = tf.Variable(tf.zeros([FLAGS.buffer_size, FLAGS.episode_len], dtype=tf.float32),
        trainable=False, name="nd_replay")
    ix0 = tf.mod(episode_num, FLAGS.buffer_size)
    ix1 = tf.Variable(0,dtype=tf.int32, trainable=False)
    ix = [[ix0,ix1]]
  add_state = tf.scatter_nd_update(ss, ix, [tf.placeholder(tf.float32,name="s")])
  with tf.control_dependencies([add_state,
    tf.scatter_update(lens, [ix0], [ix1]),
    tf.scatter_nd_update(aa, ix, [tf.placeholder(tf.int32,name="a")]),
    tf.scatter_nd_update(nd, ix, [tf.placeholder(tf.float32,name="nd")]),
    tf.scatter_nd_update(rr, ix, [tf.placeholder(tf.float32,name="r")])]):
    tf.assign_add(ix1, 1, name="add_experience")
  with tf.control_dependencies([add_state,
    tf.scatter_update(lens, [ix0], [ix1])]):
    tf.group(tf.assign_add(episode_num, 1), tf.assign(ix1, 0), name="end_episode")
  with tf.variable_scope("batch"):
    i = tf.random_uniform([n_ep], maxval=FLAGS.buffer_size, dtype=tf.int32)
    len_samples = tf.gather(lens, i, name="lens")
    trace_sizes = tf.minimum(n_exp, len_samples, name="sizes")
    j = tf.map_fn(partial(random_trace, n_exp), (len_samples,trace_sizes),
        dtype=tf.int32, back_prop=False)
    samples = tf.stack([tf.tile(tf.expand_dims(i, 1), [1, n_exp]), j], axis=2, name="samples")
    actions = tf.gather_nd(aa, samples, name="actions")
    rewards = tf.gather_nd(rr, samples, name="rewards")
    observations = tf.gather_nd(ss, samples, name="obs")
    new_observations = tf.gather_nd(ss, samples + [0,1], name="new_obs")
    not_done = tf.expand_dims(tf.gather_nd(nd, samples, name="not_done"), -1)
  return actions, rewards, observations, new_observations, not_done, trace_sizes

def model(env):
  step = tf.Variable(0, trainable=False, name='global_step')
  episode_num = tf.Variable(0,dtype=tf.int32,name='episode_num',trainable=False)
  eps = exploration_param()
  n_ep = tf.placeholder(tf.int32,name="n_ep")
  n_exp = tf.placeholder(tf.int32,name="n_exp")
  actions, rewards, observations, new_observations, not_done, lens = \
      experience_replay(env, episode_num, n_ep, n_exp) 
  with tf.variable_scope("main"): build_net(env, eps, n_ep, n_exp, observations, lens)
  with tf.variable_scope("chooser"): build_net(env, eps, n_ep, n_exp, new_observations, lens)
  with tf.variable_scope("target"): build_net(env, eps, n_ep, n_exp, new_observations, lens)
  tf.group(*[dst.assign(src) for src, dst in zip(
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main'),
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target'))], name="update_target")
  tf.group(*[dst.assign(src) for src, dst in zip(
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main'),
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'chooser'))], name="update_chooser")
  policy_onehot = tf.one_hot(ref("chooser/greedy:0"), 2, dtype=tf.float32)
  nextQ = tf.reduce_sum(tf.multiply(ref("target/qvals:0"), policy_onehot), axis=-1, name="nextQ")
  targetQ = tf.stop_gradient(rewards + FLAGS.gamma * not_done * nextQ, name="targetQ")
  actions_onehot = tf.one_hot(actions, 2, dtype=tf.float32)
  predictedQ = tf.reduce_sum(tf.multiply(ref("main/qvals:0"), actions_onehot), axis=-1, name="predictedQ")
  td_err = tf.subtract(targetQ, predictedQ, name="td_err") #b x t x a
  inbounds_mask = tf.cast(tf.range(n_exp) < tf.expand_dims(lens, 1), tf.float32)
  latter_mask = tf.cast(tf.range(n_exp) >= tf.div(n_exp, 2), tf.float32)
  masked_err = tf.multiply(tf.expand_dims(inbounds_mask * latter_mask, 2), td_err, name="masked_err")
  loss = tf.divide(tf.reduce_sum(tf.square(masked_err)),
      tf.cast(tf.reduce_sum(lens), tf.float32), name="loss")
  opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  # main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main')
  # grads,_ = tf.clip_by_global_norm(tf.gradients(loss, main_vars), 40.0)
  # opt.apply_gradients(zip(grads, main_vars), name="train", global_step=step)
  opt.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main'),
      name="train", global_step=step)
  tf.summary.scalar("max_predicted_q", tf.reduce_max(predictedQ))
  tf.summary.scalar("loss_val", loss)
  tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), name="desc")

def epoch(sess, env, cmd):
  rnn = None
  obs = env.reset()
  for t in range(FLAGS.episode_len):
    fd = {"batch/obs:0":[[obs]],'n_exp:0':1,'n_ep:0':1}
    if rnn is not None: fd['main/state_in:0'] = rnn  
    a,rnn = sess.run([cmd, "main/state_out:0"], fd)
    new_obs, reward, done, _ = env.step(a[0,0])
    yield t,obs,a[0,0],reward,new_obs,done
    if done: break
    obs = new_obs

def train_model(sess, dbg, writer, save, env):
  episode_num, step = sess.run(["episode_num:0", "global_step:0"])
  fd = {'n_exp:0': FLAGS.trace_size, 'n_ep:0': FLAGS.batch_size}
  sess.run("update_chooser")
  sess.run("update_target")
  try:
    while FLAGS.total_episodes is None or episode_num < FLAGS.total_episodes:
      episode_num = sess.run("episode_num:0")
      for (t,s,a,r,s1,d) in epoch(sess, env, "main/explore:0"):
        sess.run("add_experience", feed_dict={'a:0':a,'s:0':s,'r:0':r,'nd:0': not d})
        if episode_num >= FLAGS.buffer_size - 1 and (t % FLAGS.train_rate) == 0:
          if step % FLAGS.summary_rate == 0:
            _,smry = sess.run(["train", "desc/desc:0"], fd)
            writer.add_summary(smry, global_step=step)
          else:
            sess.run("train", feed_dict=fd)
          step = sess.run("global_step:0")
          sess.run("update_chooser")
          if step % FLAGS.target_update_rate == 0:
            sess.run("update_target")
      sess.run("end_episode", feed_dict={'s:0': s1})
      sess.run("dec_eps")
      if episode_num % FLAGS.validate_rate == 0:
        rew = validate(sess, env)
        print("Reward", rew)
        smry = sess.run("avg_r_summary:0", feed_dict={"avg_r:0":rew})
        writer.add_summary(smry, global_step=step)
      if episode_num % FLAGS.save_rate == 0:
        save(global_step=step)
  finally:
    save(global_step=step)

def validate(sess, env):
  return episode_reward(epoch(sess, env, "main/greedy:0"))

def run(env_f):
  handle_modes(env_f, model, validate, train_model)
