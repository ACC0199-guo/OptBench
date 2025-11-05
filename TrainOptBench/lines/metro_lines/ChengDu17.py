from TrainOptBench.lines.BaseSection import BaseSection


class Section1(BaseSection):
    def __init__(self):
        super().__init__()
        self.S_STA: str = "JiTouQiao"  # 出发站
        self.E_STA: str = "BaiFuoQiao"  # 到达站
        self.S_LEN: float = 1470.0  # 站间长度
        self.D_DIS: float = 10.0  # 位置离散
        self.SCH_TIME: float = 86.0  # 计划运行时间
        self.S_S_LIM: dict[int, int] = {  # 线路限速
            0: 80, 240: 120, 1025: 80, 1470: 0
        }
        self.S_SLO: dict[int, float] = {  # 坡度
            0: 0, 209: -22, 459: -5, 709: 5.47, 959: 26, 1234: 0
        }
        self.S_CUR: dict[int, float] = {  # 曲率
            0: 0, 850: 2500, 1065: 0, 1107: 1500, 1303: 0
        }
        self.S_DIR: str = "Up"  # 运行方向, up or down
        self.S_T_POWER: float = 51.56  # 牵引能耗
        self.S_R_POWER: float = 25.76  # 再生制动产生能量
        self.S_A_POWER: float = 25.80  # 实际能耗
        self.GATE_POINT: dict[int, float] = {  # 门控点

        }
        self.LINE_NAME: str = "Default"
        self.ALPHA_GATE: float = 0.0
        self.GATE_MODE: bool = False


class Section2(BaseSection):
    def __init__(self):
        super().__init__()
        self.S_STA: str = "HRD"  # 出发站
        self.E_STA: str = "HGD"  # 到达站
        self.S_LEN: float = 22149.0  # 站间长度
        self.D_DIS: float = 40.0  # 位置离散
        self.SCH_TIME: float = 780.0  # 计划运行时间
        self.S_S_LIM: dict[int, int] = {  # 线路限速
            0: 180, 21321: 80, 22149: 0
        }
        self.S_SLO: dict[int, float] = {  # 坡度
            0: 0, 4025: 0, 20000: 0
        }
        self.S_CUR: dict[int, float] = {  # 曲率
            0: 0, 22149: 0
        }
        self.S_DIR: str = "Up"  # 运行方向, up or down
        self.S_T_POWER: float = 92.94  # 牵引能耗
        self.S_R_POWER: float = 58.64  # 再生制动产生能量
        self.S_A_POWER: float = 34.3  # 实际能耗
        self.GATE_POINT: dict[int, float] = {  # 门控点
            500: 80, 1000: 120, 2000: 160, 3000: 160, 4000: 160, 5000: 160, 6000: 160, 7000: 160, 8000: 160,
            9000: 160, 10000: 160,
            11000: 160, 12000: 160, 13000: 160,
            14000: 160, 15000: 160, 16000: 160, 17000: 160, 18000: 160, 19000: 160, 20000: 160, 21000: 80,
            21321: 60, 22149: 0.1
        }
        self.TIME_POSITION_DICT: dict[int, float] = {}
        self.TIME_SPEED_DICT: dict[int, float] = {}
        self.LINE_NAME: str = "Down"
        self.ALPHA_GATE: float = 0.5
        self.GATE_MODE: bool = True
        self.calculate_train_profile()

    def calculate_train_profile(self):
        """
        计算列车匀加速、匀速、匀减速三阶段的速度、位置、时间关系。

        返回:
            tuple: 包含两个字典 (position_speed_dict, time_speed_dict)
        """
        # 已知参数
        a_acc = 1  # m/s^2
        a_dec = -1  # m/s^2
        v_max = 44.4  # m/s
        s_total = 22150.0  # m
        dt = 1.0  # 时间分辨率 1s

        # 阶段1：加速
        t_acc = v_max / a_acc
        s_acc = 0.5 * a_acc * t_acc ** 2

        # 阶段3：减速
        t_dec = (0 - v_max) / a_dec
        s_dec = v_max * t_dec + 0.5 * a_dec * t_dec ** 2

        # 阶段2：匀速
        s_const = s_total - s_acc - s_dec
        t_const = s_const / v_max
        # 初始化字典
        time_position_dict = {}
        time_speed_dict = {}

        # 初始化状态
        t = 0.0
        s = 0.0
        v = 0.0

        # 模拟过程
        total_simulation_time = t_acc + t_const + t_dec
        while t <= total_simulation_time + dt:  # 稍微多跑一点确保终点被记录
            # 确定当前阶段并计算速度和位置
            if t <= t_acc:
                # 加速阶段
                v = a_acc * t
                s = 0.5 * a_acc * t ** 2
            elif t <= t_acc + t_const:
                # 匀速阶段
                elapsed_t_const = t - t_acc
                v = v_max
                s = s_acc + v_max * elapsed_t_const
            elif t <= t_acc + t_const + t_dec:
                # 减速阶段
                elapsed_t_dec = t - t_acc - t_const
                v = v_max + a_dec * elapsed_t_dec
                s = s_acc + s_const + v_max * elapsed_t_dec + 0.5 * a_dec * elapsed_t_dec ** 2
            else:
                # 运行结束
                v = 0.0
                s = s_total

            # 记录数据，键为整数
            # 为避免浮点数精度问题导致的键重复，我们对时间和位置进行四舍五入
            time_key = int(round(t))

            # 更新字典（如果键已存在，则会被新值覆盖，但对于dt=1s的情况，通常不会）
            # 为了确保记录的是该整数秒或整数米处的速度，我们直接赋值
            time_speed_dict[time_key] = round(v, 2)
            time_position_dict[time_key] = round(s, 2)

            # 更新时间
            t += dt

            # 如果已经到达终点，可以提前结束循环
            if s >= s_total and v <= 0:
                break
        self.TIME_POSITION_DICT = time_position_dict
        self.TIME_SPEED_DICT = time_speed_dict


Section = {"Section1": Section1(), "Section2": Section2()}
