import gym
import numpy as np
import tensorflow as tf
from functools import partial

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('elite_frac', 0.2, 'Portion of highest samples to keep')
flags.DEFINE_integer('sample_size', 40, 'Number of samples each round')
flags.DEFINE_integer('n_iter', 20, 'Number of times to sample')
flags.DEFINE_integer('num_tries', 1, 'Number of rollouts per sample')

def cem(f, th_mean, initial_std=10.0):
  n_elite = int(np.round(FLAGS.sample_size*FLAGS.elite_frac))
  th_std = np.ones_like(th_mean) * initial_std
  for i in range(FLAGS.n_iter):
    ths = np.random.randn(FLAGS.sample_size, *th_mean.shape) * th_std + th_mean
    ys = np.array([f(th) for th in ths])
    elite_inds = ys.argsort(axis=0)[-n_elite:]
    if len(elite_inds.shape) > 1:
      elite_ths = ths[np.expand_dims(elite_inds,1), np.expand_dims(np.arange(ths.shape[1]), 1),
          np.arange(ths.shape[2])]
    else:
      elite_ths = ths[elite_inds]
    th_mean = elite_ths.mean(axis=0)
    th_std = elite_ths.std(axis=0)
    m = ys.mean(axis=0)
    print(np.sum(m))
  return (m, th_mean, th_std)

def noisy_evaluation(env, theta, tries=1, steps=200):
  total_rew = 0
  for _ in range(tries):
    ob = env.reset()
    for t in range(steps):
      a = (ob.reshape(-1).dot(theta) < 0).astype(np.int8)
      ob, reward, done, _ = env.step(a)
      total_rew += reward
      if done: break
  return total_rew / tries

def run(env_f):
  env = env_f()
  try: shape = (np.prod(env.observation_space.shape), env.action_space.shape)
  except: shape = env.observation_space.shape
  result, th_mean, th_std = cem(partial(noisy_evaluation, env,
    tries=FLAGS.num_tries, steps=FLAGS.episode_len), np.zeros(shape))
  # print("th mean", th_mean)
