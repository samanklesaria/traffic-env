from gym.envs.registration import register
import gym

# Frame skipping must allow rendering of internal steps.
# Just set 'rendering=True' for the level at which you want to render
def patched_step(self, action):
  if self.rendering: self.render()
  return self._step(action)
gym.Env.rendering = False
gym.Env.step = patched_step

register(
    id='traffic-v0',
    entry_point='gym_traffic.envs:TrafficEnv',
)

register(
    id='continuous-cooling-v0',
    entry_point='gym_traffic.envs:ContinuousCool',
)

register(
    id='discrete-cooling-v0',
    entry_point='gym_traffic.envs:DiscreteCool',
)




