from gym.envs.registration import register
import gym

# Frame skipping must allow rendering of internal steps.
# Just set 'rendering=True' for the level at which you want to render
def patched_step(self, action):
  if self.rendering: self.render()
  return self._step(action)
gym.Env.rendering = False
gym.Env.step = patched_step

# Traffic env has a non-scalar reward. We need envs to know the shape of their rewards.
gym.Env.reward_size = 1
prev_init = gym.Wrapper.__init__
def patched_init(self, env):
  prev_init(self, env)
  self.reward_size = env.reward_size
gym.Wrapper.__init__ = patched_init

register(
    id='traffic-v0',
    entry_point='gym_traffic.envs:TrafficEnv',
)
