import gym
import numpy as np

# This should also allow you to sum over a particular axis
def StrobeWrapper(repeat_count, num_samples, sum_axes=[]):
    class StrobeWrapper(gym.Wrapper):
        def __init__(self, env):
            super(StrobeWrapper, self).__init__(env)
            self.repeat_count = repeat_count
            self.sample_size = self.repeat_count // num_samples
            assert self.sample_size * num_samples == self.repeat_count
            self.observation_space = env.observation_space.replicated(num_samples)
            self.history = self.observation_space.empty()
            self.mask = np.zeros_like(env.observation_space.limit)
            self.mask[sum_axes] = 1

        def _step(self, action):
            done = False
            total_reward = 0
            current_step = 0
            for current_step in range(self.repeat_count):
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                if (current_step % self.sample_size) == self.sample_size - 1:
                  self.history[current_step // self.sample_size] = obs
                else:
                  self.history[current_step // self.sample_size] += (obs * self.mask)
                if done: return self.history[:(current_step + 1) //
                  self.sample_size], total_reward, done, info
            return self.history, total_reward, done, info

        def _reset(self):
            self.env.reset()
            return self.step(self.env.action_space.sample())[0]

    return StrobeWrapper


def LastWrapper(repeat_count):
    class LastWrapper(gym.Wrapper):
        def __init__(self, env):
            super(LastWrapper, self).__init__(env)

        def _step(self, action):
            done = False
            total_reward = 0
            current_step = 0
            for current_step in range(repeat_count):
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
            return obs, total_reward, done, info

        def _reset(self): return self.env.reset()
    return LastWrapper

