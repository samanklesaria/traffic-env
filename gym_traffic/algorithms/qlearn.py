import tensorflow as tf
from gym_traffic.algorithms.util import *

add_argument('--beta', 0.001, type=float)

def qlearn_derivations():
  if FLAGS.trainer == 'qlearn':
    FLAGS.history = 10
add_derivation(qlearn_derivations)

def build_net(env, temp, observations):
  reshaped = tf.reshape(observations, [-1, env.observation_space.shape[1]])
  h0 = tf.layers.dense(reshaped, 200, tf.nn.relu)
  h1 = tf.layers.dense(h0, 220, tf.nn.relu)
  together = h1.reshape(-1, env.observation_space.size)
  flat_qval = tf.layers.dense(together, env.action_space.size * 2, name="qout")
  qvals = tf.reshape(flat_qval, (-1, env.action_space.size, 2), name="qvals")
  softmax_decision(qvals, temp)

def exp_replay(env):
  with tf.variable_scope("replay"):
    a = tf.Variable(tf.zeros((FLAGS.buffer_size, env.action_space.size),
      dtype=tf.int32), trainable=False, name="action_replay")
    r = tf.Variable(tf.zeros([FLAGS.buffer_size],
      dtype=tf.float32), trainable=False, name="reward_replay")
    d = tf.Variable(tf.zeros(FLAGS.buffer_size, dtype=tf.float32), trainable=False, name="nd_replay")
    s = tf.Variable(tf.zeros((FLAGS.buffer_size, *env.observation_space.shape),
      dtype=tf.float32), trainable=False, name="obs_replay")
    s1 = tf.Variable(tf.zeros((FLAGS.buffer_size, *env.observation_space.shape),
      dtype=tf.float32), trainable=False, name="new_obs_replay")
    exp_idx = tf.Variable(0,dtype=tf.int32, trainable=False, name="exp_idx")
    mod_exp_idx = tf.reshape(tf.mod(exp_idx, FLAGS.buffer_size), [1])
  with tf.control_dependencies([
    tf.scatter_update(a, mod_exp_idx, tf.expand_dims(tf.placeholder(tf.int32,name="a"),0)),
    tf.scatter_update(r, mod_exp_idx, tf.expand_dims(tf.placeholder(tf.float32,name="r"),0)),
    tf.scatter_update(d, mod_exp_idx, tf.expand_dims(tf.placeholder(tf.float32,name="d"),0)),
    tf.scatter_update(s, mod_exp_idx, tf.expand_dims(tf.placeholder(tf.float32,name="s"),0)),
    tf.scatter_update(s1, mod_exp_idx, tf.expand_dims(tf.placeholder(tf.float32,name="s1"),0))]):
    tf.assign_add(exp_idx, 1, name="add_experience")
  with tf.variable_scope("batch"):
    samples = tf.random_uniform([tf.placeholder(tf.int32,name="n")],
        maxval=FLAGS.buffer_size, dtype=tf.int32, name="samples")
    actions = tf.gather(a, samples, name="actions")
    rewards = tf.gather(r, samples, name="rewards")
    observations = tf.gather(s, samples, name="obs")
    new_observations = tf.gather(s1, samples, name="new_obs")
    notdone = tf.expand_dims(-(tf.gather(d, samples, name="done") - 1), 1)
  return actions, rewards, observations, new_observations, notdone

def model(env):
  step = tf.Variable(0, trainable=False, name='global_step')
  episode_num = tf.Variable(0,dtype=tf.int32,name='episode_num',trainable=False)
  tf.assign_add(episode_num, 1, name="incr_episode")
  eps = exploration_param()
  actions, rewards, observations, new_observations, notdone = exp_replay(env)
  with tf.variable_scope("main"): build_net(env, eps, observations)
  with tf.variable_scope("chooser"): build_net(env, eps, new_observations)
  with tf.variable_scope("target"): build_net(env, eps, new_observations)
  tf.group(*[dst.assign(src) for src, dst in zip(
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main'),
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target'))], name="update_target")
  tf.group(*[dst.assign(src) for src, dst in zip(
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main'),
    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'chooser'))], name="update_chooser")
  policy_onehot = tf.one_hot(ref("chooser/greedy:0"), 2, dtype=tf.float32)
  nextQ = tf.reduce_sum(tf.multiply(ref("target/qvals:0"), policy_onehot), axis=2)
  targetQ = tf.stop_gradient(rewards + FLAGS.gamma * notdone * nextQ, name="targetQ")
  actions_onehot = tf.one_hot(actions, 2, dtype=tf.float32)
  predictedQ = tf.reduce_sum(tf.multiply(ref("main/qvals:0"), actions_onehot),
      axis=2, name="predictedQ")
  diff = tf.subtract(targetQ, predictedQ, name="diff")
  rho_update = []
  if FLAGS.use_avg:
    on_policy = tf.cast(tf.equal(actions, ref("main/greedy:0")), tf.float32, name="on_policy")
    num_on_policy = tf.reduce_sum(on_policy)
    tf.summary.scalar("num_on_policy", num_on_policy)
    rho_update = [tf.assign_add(rho, FLAGS.beta * tf.reduce_sum(on_policy * diff) / num_on_policy)]
  loss = tf.reduce_mean(tf.square(targetQ - predictedQ), name="td_err")
  opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  # train = opt.minimize(loss, var_list=tf.get_collection(
  #   tf.GraphKeys.TRAINABLE_VARIABLES, 'main'), global_step=step)
  main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main')
  grads,_ = tf.clip_by_global_norm(tf.gradients(loss, main_vars), 10.0)
  train = opt.apply_gradients(zip(grads, main_vars), global_step=step)
  tf.group(train, *rho_update, name="train")
  tf.summary.scalar("max_predicted_q", tf.reduce_max(predictedQ))
  tf.summary.scalar("loss", loss)
  tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES), name="desc")
  tf.summary.scalar("avg_r_summary", tf.placeholder(tf.float32, name="avg_r"))

def epoch(sess, env, cmd):
  obs = env.reset()
  for t in range(FLAGS.episode_len):
    a, = sess.run(cmd, {'batch/obs:0':[obs], 'batch/n:0':1})
    new_obs, reward, done, info = env.step(a)
    yield t,obs,a,reward,info,new_obs,done
    if done: break
    obs = new_obs

def train_model(sess, dbg, writer, save, save_best, env):
  episode_num, step = sess.run(["episode_num:0", "global_step:0"])
  sess.run("update_chooser")
  sess.run("update_target")
  best_threshold = FLAGS.best_threshold
  try:
    while FLAGS.total_episodes is None or episode_num < FLAGS.total_episodes:
      episode_num = sess.run("episode_num:0")
      for (t,s,a,r,_,s1,d) in epoch(sess, env, "main/explore:0"):
        ix = sess.run("replay/exp_idx:0")
        sess.run("add_experience", feed_dict={'a:0':a,'s:0':s,'s1:0':s1,'r:0':r,'d:0':d})
        if ix >= FLAGS.buffer_size and (ix % FLAGS.train_rate) == 0:
          if step % FLAGS.summary_rate == 0:
            _,smry = dbg.run(["train","desc/desc:0"], feed_dict={'batch/n:0':FLAGS.batch_size})
            writer.add_summary(smry, step)
          else:
            dbg.run("train", feed_dict={'batch/n:0':FLAGS.batch_size})
          step = sess.run("global_step:0")
          sess.run("update_chooser")
        if step % FLAGS.target_update_rate == 0:
          sess.run("update_target")
      sess.run("incr_episode")
      sess.run("dec_eps")
      if episode_num % FLAGS.validate_rate == 0:
        rew = validate(sess, env)[0]
        print("Reward", rew)
        smry = sess.run("avg_r_summary:0", feed_dict={"avg_r:0":rew})
        writer.add_summary(smry, episode_num)
        if best_threshold < rew:
          save_best(global_step=step)
          best_threshold = rew
      if episode_num % FLAGS.save_rate == 0:
        save(global_step=step)
  finally:
    if step is not None: save(global_step=step)

def validate(sess, env):
  return episode_reward(env, epoch(sess, env, "main/greedy:0"))

def run(env_f):
  return handle_modes(env_f, model, validate, train_model)
