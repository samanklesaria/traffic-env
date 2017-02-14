import gym
import numpy as np
import tensorflow as tf
from functools import partial
import json

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('elite_frac', 0.06, 'Portion of highest samples to keep')
flags.DEFINE_integer('sample_size', 60, 'Number of samples each round')
flags.DEFINE_integer('n_iter', 100, 'Number of times to sample')
flags.DEFINE_integer('num_tries', 1, 'Number of rollouts per sample')
flags.DEFINE_boolean('restore_cem', True, 'Whether to restore the model')

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
    print(np.mean(m))
    yield th_mean

def noisy_evaluation(env, theta, tries=1, steps=200):
  total_rew = 0
  for _ in range(1):
    ob = env.reset()
    multiplier = 1.0
    for t in range(steps):
      a = (ob.reshape(-1).dot(theta) < 0).astype(np.int8)
      ob, reward, done, _ = env.step(a)
      total_rew += reward * (multiplier if FLAGS.print_discounted else 1)
      multiplier *= FLAGS.gamma
      if done: break
  return total_rew / tries

def run(env_f):
  env = env_f()
  shape = (env.observation_space.size, env.action_space.size)
  try:
    with open('weights.json', 'r') as f:
      weights = np.reshape(np.array(json.load(f)), shape)
  except: weights = np.zeros(shape)
  th_mean = weights
  try:
    for weights in cem(partial(noisy_evaluation, env,
      tries=FLAGS.num_tries, steps=FLAGS.episode_len), weights):
      th_mean = weights
  except KeyboardInterrupt:
    pass
  with open('weights.json', 'w') as f:
    json.dump(np.reshape(th_mean, [*env.observation_space.shape,
      *env.action_space.shape]).tolist(), f, indent=4,  separators=(',', ': '))
  print("Saved to weights.json")
