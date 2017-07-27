import gym

def WarmupWrapper(ignore_count):
  class WarmupWrapper(gym.Wrapper):
    def __init__(self, env):
      super(WarmupWrapper, self).__init__(env)
      self.ignore_count = ignore_count
    def _reset(self):
      while True:
        obs = self.env.reset()
        for _ in range(self.ignore_count):
          obs, _, done, _ = self.env.step(self.env.action_space.sample())
          if done:
            print("Episode completed during warmup")
            break
        else: return obs
  return WarmupWrapper
