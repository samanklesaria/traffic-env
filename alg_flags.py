import importlib
from args import add_argument, add_derivation, FLAGS, PARSER

add_argument('--restore', False, type=bool)
add_argument('--grad_summary', False, type=bool)
add_argument('--reward_printing', 'discounted')
add_argument('--render', False, type=bool)
add_argument('--episode_len', 500, type=int)
add_argument('--save_rate', 1000, type=int)
add_argument('--logdir', 'summaries')
add_argument('--gamma', 0.99, type=float)
add_argument('--learning_rate', 0.001, type=float)
add_argument('--summary_rate', 5, type=int)
add_argument('--validate_rate', 25, type=int)
add_argument('--trainer', "qlearn")
add_argument('--exploration', "e_greedy")
add_argument('--batch_size', 30, type=int)
add_argument('--vis_size', 200, type=int)
add_argument('--mode', 'train')
add_argument('--spacing', 3, type=int)
add_argument('--start_eps', 0.8, type=float)
add_argument('--end_eps', 0.01, type=float)
add_argument('--start_temp', 0.2, type=float)
add_argument('--end_temp', 0.01, type=float)
add_argument('--annealing_episodes', 10000, type=float)
add_argument('--history', 1, type=int)
add_argument('--target_update_rate', 50, type=int)
add_argument('--buffer_size', 1000, type=int)
add_argument('--trace_size', 8, type=int)
add_argument('--threads', 4, type=int)
add_argument('--lam', 1, type=float)
add_argument('--debug', False, type=bool)
add_argument('--train_rate', 1, type=int)
add_argument('--total_episodes', None, type=int)

def standard_derivations():
  FLAGS.print_discounted = FLAGS.reward_printing == "discounted"
  FLAGS.print_avg = FLAGS.reward_printing == "avg"
  if FLAGS.render: FLAGS.threads = 1
add_derivation(standard_derivations)

def run_alg(env_f):
  importlib.import_module("gym_traffic.algorithms."+FLAGS.trainer).run(env_f)
