import gym
from gym_traffic.wrappers.gspace import GSpaceWrapper
import numpy as np
from args import parse_flags, update_flags, FLAGS
from alg_flags import run_alg

def make_env():
  return GSpaceWrapper(gym.make('CartPole-v0'))

if __name__ == '__main__':
  parse_flags()
  update_flags(
    learning_rate = 0.1,
    episode_len = 800,
    gamma = 0.99,
    summary_rate = 20,
    save_rate = 10000,
    train_rate = 1,
    batch_size = 10,
    target_update_rate = 5,
    annealing_episodes = 1000,
    buffer_size = 50,
    lam = 1,
    start_eps = 0.2,
    min_eps = 0.01,
    print_discounted = False,
    reward_printing = "sum",
    trace_size = 1,
    validate_rate = 20)
  run_alg(make_env)
