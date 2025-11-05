class BaseSection:
    def __init__(self):
        self.S_STA: str = "Default"  # 出发站
        self.E_STA: str = "Default"  # 到达站
        self.S_LEN: float = 0.0  # 站间长度
        self.D_DIS: float = 0.0  # 位置离散
        self.SCH_TIME: float = 0.0  # 计划运行时间
        self.S_S_LIM: dict[int, int] = {  # 线路限速

        }
        self.S_SLO: dict[int, float] = {  # 坡度

        }
        self.S_CUR: dict[int, float] = {  # 曲率

        }
        self.S_DIR: str = "Default"  # 运行方向, up or down
        self.S_T_POWER: float = 0.0  # 牵引能耗
        self.S_R_POWER: float = 0.0  # 再生制动产生能量
        self.S_A_POWER: float = 0.0  # 实际能耗
        self.GATE_POINT: dict[int, int] = {  # 门控点

        }
        self.LINE_NAME: str = "Default"
        self.ALPHA_GATE: float = 0.0
        self.GATE_MODE: bool = False
