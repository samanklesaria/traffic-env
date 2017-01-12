import gym
import numpy as np

def StrobeWrapper(repeat_count, num_samples):
    class StrobeWrapper(gym.Wrapper):
        def __init__(self, env):
            super(StrobeWrapper, self).__init__(env)
            self.repeat_count = repeat_count
            self.sample_size = self.repeat_count // num_samples
            assert self.sample_size * num_samples == self.repeat_count
            self.history = np.empty((num_samples, *self.env.observation_space.shape), dtype=np.float32)
            self.observation_space = gym.spaces.Box(
              np.tile(self.env.observation_space.low, (num_samples, 1)),
              np.tile(self.env.observation_space.high, (num_samples, 1)))

        def _step(self, action):
            done = False
            total_reward = 0
            current_step = 0
            for current_step in range(self.repeat_count):
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                if (current_step % self.sample_size) == self.sample_size - 1:
                  self.history[current_step // self.sample_size] = obs
                if done: return self.history[:(current_step + 1) //
                    self.sampe_size], reward, done, info
            return self.history, total_reward, done, info

        def _reset(self):
            self.env.reset()
            return self.step(self.env.action_space.sample())[0]

    return StrobeWrapper
