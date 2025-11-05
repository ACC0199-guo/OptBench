import numpy as np
from .BaseMetroEnv import BaseMetroEnv


class CustomMetroEnv(BaseMetroEnv):
    def __init__(self, config, reward_type="A"):
        super().__init__(config)
        self.reward_type = reward_type

    def calc_reward(self):
        # 四种奖励方式，根据 self.reward_type 选择
        if self.reward_type == "A":
            GATE_REWARD = 0
            PUNISH_REWARD = 0
            END_REWARD = 0
            if hasattr(self.sim_train.SECTION, 'GATE_MODE') and self.sim_train.SECTION.GATE_MODE:
                d = - abs(self.sim_train.N_L / self.sim_train.GATE_LOCATION - 1)
                v = - abs(self.sim_train.N_S / (self.sim_train.GATE_SPEED / 3.6) - 1)
                v = max(v, -1)
                d = max(d, -1)
                if abs(self.sim_train.N_L - self.sim_train.GATE_LOCATION) <= 50 and abs(self.sim_train.N_S - self.sim_train.GATE_SPEED / 3.6) <= 3.6:
                    r = 10
                else:
                    r = 0
                GATE_REWARD = d + 1 * v + r
                if self.sim_train.OVER_SPEED:
                    PUNISH_REWARD = -20
                if self.sim_train.N_L < (min(self.sim_train.ATP_LIM, self.sim_train.STA_LIM) / 3.6 - 10.8) and 0.1 * self.sim_train.SECTION.S_LEN <= self.sim_train.N_L <= 0.985 * self.sim_train.SECTION.S_LEN:
                    PUNISH_REWARD = -10
                if min(self.sim_train.ATP_LIM, self.sim_train.STA_LIM) / 3.6 > self.sim_train.N_S >= (
                        min(self.sim_train.ATP_LIM, self.sim_train.STA_LIM) / 3.6 - 10.8) and 0.1 * self.sim_train.SECTION.S_LEN <= self.sim_train.N_L <= 0.985 * self.sim_train.SECTION.S_LEN:
                    PUNISH_REWARD = 0
                if self.sim_train.C_L >= self.sim_train.SECTION.S_LEN:
                    END_REWARD = 0.33 * abs(self.sim_train.C_L - self.sim_train.SECTION.S_LEN) + abs(self.sim_train.C_S - 0) + abs(self.sim_train.C_T - self.sim_train.SECTION.SCH_TIME)
                return GATE_REWARD + PUNISH_REWARD - 0 * END_REWARD
            else:
                raise ValueError("没有门控点")
        elif self.reward_type == "B":
            # 示例：距离终点越近奖励越高
            if self.sim_train.SECTION is not None and hasattr(self.sim_train.SECTION, "S_LEN"):
                return -abs(self.sim_train.SECTION.S_LEN - self.sim_train.C_L)
            else:
                return 0  # 或者根据实际需求返回其他默认值
        elif self.reward_type == "C":
            # 示例：奖励为当前速度
            return self.sim_train.C_S
        elif self.reward_type == "D":
            # 示例：奖励为负能耗（假设有能耗属性）
            return -getattr(self.sim_train, "energy", 0)
        else:
            raise ValueError("未知的reward_type")

    def is_terminated(self):
        self.DONE = False
        # 终止条件示例：到达终点
        if abs(self.sim_train.C_L - self.sim_train.SECTION.S_LEN) <= 5:
            self.DONE = True
            return True
        else:
            return False
