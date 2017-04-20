import tensorflow as tf
import numpy as np
from util import print_running_stats
from gym_traffic.envs.traffic_env import cars_on_roads
from args import FLAGS

def run(env_f,_):
  FLAGS.learn_switch = False
  env = env_f()
  def rewards():
    while True:
      multiplier = 1
      total_reward = 0
      env.reset()
      action = np.zeros(env.action_space.shape)
      for i in range(FLAGS.episode_len):
        obs = env.unwrapped.cars_on_roads()
        if FLAGS.render: print("OBS", obs)
        if (i % FLAGS.spacing) == 0: 
          action = env.action_space.to_action(obs.dot([1,1,-1,-1]) < 0)
        _, reward, done, _ = env.step(action)
        if FLAGS.render: print("REWARD", reward)
        total_reward += np.mean(reward) * (multiplier if FLAGS.print_discounted else 1)
        multiplier *= FLAGS.gamma
        if done: break
      yield total_reward
  print_running_stats(rewards())
