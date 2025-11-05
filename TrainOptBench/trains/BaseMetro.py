import numpy as np
from TrainOptBench.trains.BaseTrain import BaseTrain
import warnings


class BaseMetro(BaseTrain):
    """     Base single metro environment class for reinforcement learning."""

    def __init__(self, train_config):
        super().__init__(train_config)

        self.TRAIN_MODEL: str = train_config.train_model

        # 当线路的GATE_MODE = True时维护这两个变量
        self.GATE_LOCATION: float = 0.0
        self.GATE_SPEED: float = 0.0
        self.LAST_GATE: bool = False

    def _getMTF(self):
        """   The unit of speed in this function should be km/h  """
        self._getN1()
        if self.C_S * 3.6 <= self.KIN_LIST[0]:
            self.M_T_F = self.KIN_LIST[1]
        else:
            self.M_T_F = self.KIN_LIST[2] * self.C_S * 3.6 * self.C_S * 3.6 + self.KIN_LIST[3] * self.C_S * 3.6 + self.KIN_LIST[4]

    def _getMBF(self):
        """   The unit of speed in this function should be km/h  """
        self._getN1B()
        if self.C_S * 3.6 <= self.KIN_LIST[5]:
            self.M_B_F = self.KIN_LIST[6] * self.C_S * 3.6
        elif self.KIN_LIST[5] < self.C_S * 3.6 <= self.KIN_LIST[7]:
            self.M_B_F = self.KIN_LIST[8]
        else:
            self.M_B_F = self.KIN_LIST[9] * self.C_S * 3.6 + self.KIN_LIST[10]

    def _getN1(self):
        """   The unit of speed in this function should be km/h  """
        if self.C_S * 3.6 <= self.MOT_LIST[0]:
            self.N1 = self.MOT_LIST[1] * np.exp(self.MOT_LIST[2] * self.C_S * 3.6) + self.MOT_LIST[3]
        else:
            self.N1 = self.MOT_LIST[4]

    def _getN1B(self):
        # C_S需要加ERR_SPEED，电机拟合有误差，认为全部是电制动
        if 0 < self.C_S * 3.6 <= self.MOT_LIST[5]:
            self.N1_B = self.MOT_LIST[6] * np.exp(self.MOT_LIST[7] / ((self.C_S + self.ERR_SPEED) * 3.6)) + self.MOT_LIST[8]
            if self.N1_B < 0:
                self.N1_B = 0
        else:
            self.N1_B = self.MOT_LIST[9]

    def _getTPower(self):
        return self.M_T_F * self.ACTION * (self.C_S * 0.5 + self.N_S * 0.5) * (self.TIME_STEP / 3600) / (self.N1 * self.N2 * self.N3 * self.N4)

    def _getRePower(self):
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            return self.M_B_F * self.ACTION * (self.C_S * 0.5 + self.N_S * 0.5) * (self.TIME_STEP / 3600) / (self.N1_B * self.N2 * self.N3 * self.N4)
        except RuntimeWarning:
            print(f"除零错误 - 变量值: N1_B={self.N1_B}, N2={self.N2}, N3={self.N3}, N4={self.N4}")
            return 0.0  # 返回默认值避免程序崩溃

    def _getLimit(self):
        if not self.SECTION or not self.SECTION.S_S_LIM:
            self.STA_LIM = float('inf')
            self.ATP_LIM = float('inf')
            return

        next_limit_position = next_limit_value = current_limit_value = 0.0
        prev_pos = None
        for pos in sorted(self.SECTION.S_S_LIM.keys()):
            # 因为pos大于next_locate所以说明静态限速一定是前一个对应的值，即prev_pos对应的限速
            if pos > self.N_L:
                next_limit_position = pos
                next_limit_value = self.SECTION.S_S_LIM[pos]
                if prev_pos is not None:
                    current_limit_value = self.SECTION.S_S_LIM[prev_pos]
                break
            prev_pos = pos
        self.STA_LIM = current_limit_value
        if next_limit_value < current_limit_value:
            radicand = (next_limit_value / 3.6) ** 2 - 2 * self.M_B_A * (next_limit_position - self.N_L)
            self.ATP_LIM = np.sqrt(max(radicand, 0))  # 确保非负
            # self.ATP_LIM = np.sqrt(next_limit_value * next_limit_value / 3.6 / 3.6 - 2 * self.M_B_A * (next_limit_position - self.N_L))

    def _getAtpLimit(self):
        pass

    def _getStaLimit(self):
        pass

    def _getAction(self, action):
        GATE_ACTION = 0
        if self.SECTION and self.SECTION.GATE_MODE:
            GATE_ACTION = self._getGateAction()
        alpha_gate = self.SECTION.ALPHA_GATE if self.SECTION else 0.0
        if self.LAST_GATE:
            # 计算时间差，并避免除零错误
            time_diff = self.SECTION.SCH_TIME - self.C_T
            # 添加一个小的epsilon值来避免除零
            epsilon = 1e-6
            # 计算GATE_ACTION，避免除零错误
            term1 = 0.5 * (self.SECTION.S_LEN - self.C_L) / (abs(time_diff) ** 2 + epsilon)
            term2 = 0.5 * (-self.C_S / 3.6) / (time_diff + (1 if time_diff >= 0 else -1) * epsilon)
            GATE_ACTION = term1 + term2
            GATE_ACTION = max(min(GATE_ACTION, 1), -1)
        self.ACTION = (1 - alpha_gate) * action + alpha_gate * GATE_ACTION

    def _getGateAction(self):
        self._getGateData()
        GATE_ACTION = np.tanh(2 * (0.5 * (-self.C_L + self.GATE_LOCATION) / self.GATE_LOCATION - 0.5 * (self.C_S - self.GATE_SPEED / 3.6) / (self.GATE_SPEED / 3.6)) / 1)
        return GATE_ACTION
        # return np.tanh(2 * ((self.GATE_SPEED / 3.6) - self.C_S) / (self.GATE_SPEED / 3.6))

    def _getGateData(self):
        # 初始化LAST_GATE标志为False
        self.LAST_GATE = False
        if not self.SECTION or not self.SECTION.GATE_POINT:
            self.GATE_LOCATION = 0.0
            self.GATE_SPEED = 1.0
            return
        sorted_positions = sorted(self.SECTION.GATE_POINT.keys())
        prev_pos = 0
        found = False
        # 获取最后一个门控点的位置
        last_position = sorted_positions[-1] if sorted_positions else None
        for pos in sorted_positions:
            if self.C_L <= pos:
                self.GATE_LOCATION = pos
                self.GATE_SPEED = self.SECTION.GATE_POINT[pos]
                if prev_pos:
                    self.P_R_P = prev_pos
                found = True
                # 检查是否为最后一个门控点
                if pos == last_position:
                    self.LAST_GATE = True
                break
            prev_pos = pos
        # 循环结束后检查是否超过所有门控点
        if not found:
            self.GATE_LOCATION = sorted_positions[-1] if sorted_positions else self.SECTION.S_LEN
            self.GATE_SPEED = self.SECTION.GATE_POINT.get(self.GATE_LOCATION, 1)
            # 如果没有找到且存在门控点，则当前使用的是最后一个门控点
            if sorted_positions:
                self.LAST_GATE = True

    def _reshapeAction(self):
        if self.SECTION and self.C_L <= 0.05 * self.SECTION.S_LEN:
            self.ACTION = self.ACTION + 1
        else:
            self.ACTION = self.ACTION
        LOW_BOUND = -1
        UPPERBOUND = 1
        self.ACTION = np.clip(self.ACTION, LOW_BOUND, UPPERBOUND)

    def _getAcc(self):
        self._getMTF()
        self._getMBF()
        if self.ACTION < 0:
            self.T_ACC = 0
            self._getBAcc()
        else:
            self.B_ACC = 0
            self._getTAcc()
        self._getGAcc()
        self._getCuAcc()
        self._getRAcc()
        self.ACC = self.T_ACC + self.B_ACC + self.G_ACC + self.CU_ACC + self.B_R_ACC
        self.ACC = np.clip(self.ACC, self.M_B_A, self.M_T_A)
        if self.SECTION and self.C_L >= 0.99 * self.SECTION.S_LEN:
            self.ACC = max(self.ACC - 1, self.M_B_A)

    def _getTAcc(self):
        self.T_ACC = self.M_T_F * self.ACTION / self.WEIGHT

    def _getBAcc(self):
        self.B_ACC = self.M_B_F * self.ACTION / self.WEIGHT

    def _getGAcc(self):
        if not self.SECTION or not self.SECTION.S_SLO:
            self.G_ACC = 0.0
        else:
            key_list = list(self.SECTION.S_SLO.keys())
            if len(key_list) < 2:
                self.G_ACC = 0.0
            else:
                key = 0
                for j in range(len(key_list) - 1):
                    if key_list[j] <= self.C_L < key_list[j + 1]:
                        key = key_list[j]
                gradient = self.SECTION.S_SLO[key]
                self.G_ACC = -9.8 * gradient / 1000

    def _getCuAcc(self):
        if not self.SECTION or not self.SECTION.S_CUR:
            self.CU_ACC = 0.0
        else:
            key_list = list(self.SECTION.S_CUR.keys())
            if len(key_list) < 2:
                self.CU_ACC = 0.0
            else:
                key = 0
                for j in range(len(key_list) - 1):
                    if key_list[j] <= self.C_L < key_list[j + 1]:
                        key = key_list[j]
                curve = self.SECTION.S_CUR[key]
                if curve != 0:
                    self.CU_ACC = - 3 * 9.8 / (5 * curve)
                else:
                    self.CU_ACC = 0

    def _getRAcc(self):
        self.B_R_ACC = - (self.D_A + self.D_B * self.C_S * 3.6 + self.D_C * self.C_S * self.C_S * 3.6 * 3.6) / 1000

    def _getNextState(self):
        temp_time = self.C_T
        temp_speed = self.C_S
        speed = temp_speed + self.ACC * self.TIME_STEP
        if speed <= 1:
            speed = 1
        if speed * 3.6 >= self.MAX_SPEED:
            speed = self.MAX_SPEED / 3.6
        time = temp_time + self.TIME_STEP
        self.N_T = time
        self.N_S = speed
        temp_locate = self.C_L + (temp_speed * 0.5 + speed * 0.5) * self.TIME_STEP
        self.N_L = temp_locate

    def _trans2NextState(self):
        self.B_S = self.C_S
        self.B_T = self.C_T
        self.B_L = self.C_L
        self.C_S = self.N_S
        self.C_T = self.N_T
        self.C_L = self.N_L

    def _checkOverSpeed(self):
        self.OVER_SPEED = False
        if self.N_S > min(self.ATP_LIM, self.STA_LIM):
            self.OVER_SPEED = True
        else:
            self.OVER_SPEED = False

    def _getStepPower(self):
        if self.ACTION > 0:
            self.STEP_T_POWER = self._getTPower()
            self.STEP_RE_POWER = 0.0
        else:
            self.STEP_T_POWER = 0.0
            self.STEP_RE_POWER = self._getRePower()

    def _storeStepInfo(self):
        self.TRA_POWER_LIST.append(self.STEP_T_POWER)
        self.RE_POWER_LIST.append(self.STEP_RE_POWER)
        self.S_LIST.append(self.N_S)
        self.L_LIST.append(self.N_L)
        self.T_LIST.append(self.N_T)

    def execute(self, action):
        self._getAction(action)
        self._reshapeAction()
        self._getLimit()
        self._getAcc()
        self._getNextState()
        self._getStepPower()
        self._checkOverSpeed()
        self._storeStepInfo()
        self._trans2NextState()
