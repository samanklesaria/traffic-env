import gym

def WarmupWrapper(ignore_count):
  class WarmupWrapper(gym.Wrapper):
    def __init__(self, env):
      super(WarmupWrapper, self).__init__(env)
      self.ignore_count = ignore_count
    def _reset(self):
      obs = self.env.reset()
      for _ in range(self.ignore_count):
        obs, _, done, _ = self.env.step(self.env.action_space.sample())
        assert not done, "Episode completed during warmup"
      return obs
  return WarmupWrapper
