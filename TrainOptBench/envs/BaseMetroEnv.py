import numpy as np
from gymnasium import spaces
from TrainOptBench.envs.BaseSim import BaseEnv
from TrainOptBench.trains.BaseMetro import BaseMetro

from TrainOptBench.lines.metro_lines.ChengDu17 import Section


class BaseMetroEnv(BaseEnv):
    def __init__(self, config):
        # 先构造 BaseMetro 实例
        self.sim_train = BaseMetro(config)
        self.SECTION = Section[self.sim_train.SECTION_NAME]
        # 根据 ACTION_TYPE 动态定义动作空间
        if self.sim_train.ACT_TYPE == "C":
            action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        elif self.sim_train.ACT_TYPE == "D":
            action_space = spaces.Discrete(8)  # 0-7 共8个整数
        # 定义观测空间
        observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([self.sim_train.MAX_SPEED, np.inf, np.inf]),
            dtype=np.float32
        )
        self.DONE = False
        super().__init__(observation_space, action_space)
        self.config = config

    def reset(self, *, seed=None, options=None):
        self.sim_train = BaseMetro(self.config)
        self.sim_train.SECTION = self.SECTION
        obs = np.array([self.sim_train.C_S, self.sim_train.C_L, self.sim_train.C_T], dtype=np.float32)
        info = {}
        self.current_step = 0
        return obs, info

    def step(self, action):
        if self.sim_train.ACT_TYPE == "C":
            # 连续动作，action 是数组
            real_action = float(action[0])
        elif self.sim_train.ACT_TYPE == "D":
            # 离散动作，action 是整数
            real_action = int(action)
        else:
            raise ValueError("未知的ACTION_TYPE")
        self.sim_train.execute(real_action)
        obs = np.array([self.sim_train.C_S, self.sim_train.C_L, self.sim_train.C_T], dtype=np.float32)
        terminated = self.is_terminated()
        reward = self.calc_reward()
        truncated = False
        info = {}
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def calc_reward(self):
        """计算奖励，需在子类中实现"""
        raise NotImplementedError

    def is_terminated(self):
        """判断是否终止，需在子类中实现"""
        raise NotImplementedError

    def render(self):
        print(f"Step: {self.current_step}, Speed: {self.sim_train.C_S}, Pos: {self.sim_train.C_L}")
