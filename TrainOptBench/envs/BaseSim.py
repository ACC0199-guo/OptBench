import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        obs = self._get_initial_obs()
        info = {}
        return obs, info

    def step(self, action):
        obs = self._get_next_obs(action)
        reward = self._get_reward(obs, action)
        terminated = self._get_terminated(obs)
        truncated = False
        info = {}
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}")

    def close(self):
        pass

    # 下面这些方法可以在子类中重写
    def _get_initial_obs(self):
        return self.observation_space.sample()

    def _get_next_obs(self, action):
        return self.observation_space.sample()

    def _get_reward(self, obs, action):
        return 0.0

    def _get_terminated(self, obs):
        return False