import tensorflow as tf
import numpy as np
from util import *
from gym_traffic.envs.traffic_env import cars_on_roads
from args import FLAGS

def run(env_f):
  FLAGS.learn_switch = False
  env = env_f()
  prev_action = None
  def episode():
    env.reset()
    for i in range(FLAGS.episode_len):
      obs = env.unwrapped.cars_on_roads()
      if i % FLAGS.spacing == 0:
        action = env.action_space.to_action(obs.dot([1,1,-1,-1]) < 0)
      obs, reward, done, info = env.step(action)
      yield i,obs,action,reward,info
      if done: break
  data = print_running_stats(forever(lambda: episode_reward(env, episode())))
  if FLAGS.interactive: return data
  write_data(*data)
