# 开发日记

---

## 总体规划

目标：给出一个通用平台（现阶段仅针对单列车），使得能够使用stable-baseline3中的算法进行测试，并将测试结果与优化改进后的算法进行对比。
对比主要包括运行时间、精确停车、能耗这三个实际运行指标。并能够比较不同奖励函数类型、不同优化器下算法的奖励 变化情况。\
实现：使用gymnasium基类构建env，在env中可选择不同的线路、车型、奖励函数（通过parser设置）。<font color = "red">**代码设计时应先线路、再选车、再生成环境，车里内置线路（电子地图，环境把车和线路拿来实例化）** </font> 通过不同的组合构建不同的仿真环境。使用stable-baseline3构建基类对比算法。并尽可能将自身提出的算法也使用stable-baseline3来实现，并使用不同的优化器。

---

## 线路类

### 构造函数

| 描述 | 定义 | 数据类型 | 单位 |例子|
|:---: |:---: |  :---:  |:---:|:---:|
| 出发站 | S_STA | str | - | "JiTouQiao" |
| 到达站 | E_STA | str | - | "BaiFuoQiao"|
| 站间长度|S_LEN| float| m | 1470|
| 位置离散长度|D_DIS| float| m | 10|
| 计划站间运行时间| SCH_TIME | float | s | 86 |
| 站间限速 | S_S_LIM | dict | m,m/s | self.S_S_LIM = { 0: 80, 240: 120, 1025: 80, 1470: 0} |
| 站间坡度 | S_SLO   | dict | m, 千分比| self.S_SLO = { 0: 0, 209: -22, 459: -5, 709: 5.47, 959: 26, 1234: 0} |
| 站间曲率 | S_CUR   | dict | m, m  | self.S_CUR = {0: 0, 235: 800, 631: 0, 873: 3000, 1155: 0, 1462: 450, 1779: 0} |
| 站间运行方向 | S_DIR | str | - | "Up" |
| 站间牵引能耗 （计划运行时间） | S_T_POWER | float |kW$\cdot$h | 63 |
| 站间再生能量 （计划运行时间） | S_R_POWER | float |kW$\cdot$h | 21.96 |
| 站间净能耗（计划运行时间）    | S_A_POWER | float |kW$\cdot$h | 41.04 |
| 门控点列表 （可选） | GATE_POINT | dict[int,int] | m,m | - |
| 是否有门控点 | GATE_MODE | bool | - | True |
| 门控系数    | ALPHA_GATE | float | - | 0 | 
| 所在线路名称 | LINE_NAME | str | - | "ChengDu17" |
| **线路**| 直接用线路名称 | dict[str,class] |-| - |

```python
class Section1:
    def __init__(self):
        self.S_STA: str = '''JiTouQiao'''  # 出发站
        self.E_STA: str = '''BaiFuoQiao'''  # 到达站
        self.S_LEN: float = 1470  # 站间长度
        self.D_DIS: float = 10  # 位置离散
        self.SCH_TIME: float = 86  # 计划运行时间
        self.S_S_LIM: dict[int,int] = {  # 线路限速
            0: 80, 240: 120, 1025: 80, 1470: 0
        }
        self.S_SLO: dict[int,int] = {  # 坡度
            0: 0, 209: -22, 459: -5, 709: 5.47, 959: 26, 1234: 0
        }
        self.S_CUR: dict[int,int] = {0: 0, 850: 2500, 1065: 0, 1107: 1500, 1303: 0}  # 曲率
        self.S_DIR: str = "Up"  # 运行方向
        self.S_T_POWER: flaot = 51.56  # 牵引能耗
        self.S_R_POWER: float = 25.76  # 再生制动产生能量
        self.S_A_POWER: float = 25.80  # 实际能耗
        self.LINE_NAME: str = "ChengDu17"
        self.GATA_MODE: bool = False

ChengDu17 = {"Section1": Section1(), "Section2": Section2(), "Section3": Section3(),  "Section4": Section4(),
"Section5": Section5(),"Section6": Section6(), "Section7": Section7(), "Section8": Section8(),
"Section9": Section9(), "Section10": Section10(), "Section11": Section11(),"Section12": Section12(),
"Section13": Section13(), "Section14": Section14(), "Section15": Section15(),"Section16": Section16(), "Section17": Section17()}
```

 **注意事项**

 1. 站间净能耗=站间牵引能耗-站间再生能量
 2. 每一个py文件对应一个线路，应存储上下行的所有站间，该py文件下的每一个class类对应一个站间。
 3. class类定义时奇数代表上行，偶数代表下行。
 4. 如果有门控点则计算门控动作与门控奖励，否则直接取0，默认没有门控点。
 5. 每一个Section对应所在线路的区间，整个py文件构成一个线路。
 6. 门控系数默认置0

---

## 车类

*需要确定要不要把不同车的类型分开，先写一个基类，再继承*
*目前仅考虑单列车时的情况*
~~*目前仅考虑动作空间为连续的情况*~~

### 构造函数

| 描述 | 定义 | 数据类型 |单位  | 例子 | 外部获取 |
|:---: |:---: |  :---:  |:---:|:---:|:---:|
| 车重 | WEIGHT | float | t | 337.8 | Y |
| 戴维斯参数a  | D_A | float | - | 8.4 | Y |
| 戴维斯参数b  | D_B | float | - | 0.1071 | Y |
| 戴维斯参数c  | D_C | float | - | 0.00472 | Y |
| 牵引电机效率 | N1   | float | - | 0 | N |
| 制动电机效率 | N1_B | float | - | 0 | N |
| 变压器效率   | N2  | float | - | 0.9702 | Y |
| 变流器效率   | N3  | float | - | 0.96 | Y |
| 齿轮箱效率   | N4  | float | - | 0.97 | Y |
| 最大牵引力 （实时）| M_T_F | float | kN |0| N |
| 最大制动力 （实时）| M_B_F | float | kN |0| N |
| 最大牵引加速度 （能力）| M_T_A | float | $m/s^2$ | 0 | Y |
| 最大制动加速度 （能力）| M_B_A | float | $m/s^2$ | 0 | Y |
| 列车速度 （当前仿真时刻） | C_S | float | $m/s$ | 0 | N |
| 列车位置 （当前仿真时刻） | C_L | float | $m$   | 0 | N |
| 列车时间 （当前仿真时刻） | C_T | float | $s$   | 0 | N |
| 列车速度 （上一仿真时刻） | B_S | float | $m/s$ | 0 | N |
| 列车位置 （上一仿真时刻） | B_L | float | $m$ | 0 | N |
| 列车时间 （上一仿真时刻） | B_T | float | $s$ | 0 | N |
| 列车速度 （下一仿真时刻） | N_S | float | $m/s$ | 0 | N |
| 列车位置 （下一仿真时刻） | N_L | float | $m$ | 0 | N |
| 列车时间 （下一仿真时刻） | N_T | float | $s$ | 0 | N |
| 列车状态 （当前仿真时刻） | C_STATE | np.array(list[C_S,C_L,C_T]) | - | - | N |
| 策略动作 | ACTION | UNION[int,float]| - | - | - |
| 列车合加速度 | ACC | float | $m/s^2$ | 0 | N |
| 牵引加速度   | T_ACC | float | $m/s^2$ | 0 | N |
| 制动加速度   | B_ACC | float | $m/s^2$ | 0 | N |
| 坡度加速度   | G_ACC | float | $m/s^2$ | 0 | N |
| 曲率加速度   | CU_ACC | float | $m/s^2$ | 0 | N |
| 基本阻力加速度 | B_R_ACC | float | $m/s^2$ | 0 | N |
| ATP限速 | ATP_LIM | float | $m/s$ | 0 | N |
| 静态限速 | STA_LIM | float | $m/s$ | 0 | N |
| 车型限速 | MAX_SPEED | float | $m/s$ | 0 | N |
| 运行线路 | LINE | str | - | - | Y |
| 仿真区间 | SECTION | str | - | - | Y |
| 动力学特性参数组 | KIN_LIST | list[float] | - | - | Y |
| 电机效率参数组 | MOT_LIST | list[float] | - | - | Y |
| 控制类型 | ACT_TYPE | str | - | "C" | Y |

**注意事项**

1. 动力学特性参数组、电机效率参数组均为10个变量。
2. 运行线路直接使用字符串获取线路类中字典对应的区间。
3. 控制类型中"C"代表连续，"D"代表离散。

```python
class Train:
    def __init__(self):
        # 车重（吨）
        self.WEIGHT: float = 337.8
        # 戴维斯参数
        self.D_A: float = 8.4
        self.D_B: float = 0.1071
        self.D_C: float = 0.00472
        # 效率参数
        self.N1: float = 0  # 牵引电机效率
        self.N1_B: float = 0  # 制动电机效率
        self.N2: float = 0.9702  # 变压器效率
        self.N3: float = 0.96  # 变流器效率
        self.N4: float = 0.97  # 齿轮箱效率

        # 最大牵引力最大制动力
        self.M_T_F: float = 0
        self.M_B_F: float = 0

        # 最大加速度最大减速度
        self.M_T_A: float = 1.2
        self.M_B_A: float = -1.2

        # 列车的速度，位置，当前时间
        self.C_S: float = 0.0  # m
        self.C_L: float = 0.0  # m/s
        self.C_T: float = 0.0  # s
        # 列车的速度，位置，当前时间 (上一仿真时刻)
        self.B_S: float = 0.0  # m
        self.B_L: float = 0.0  # m/s
        self.B_T: float = 0.0  # s
        # 列车的速度，位置，当前时间 (下一仿真时刻)
        self.N_S: float = 0.0  # m
        self.N_L: float = 0.0  # m/s
        self.N_T: float = 0.0  # s
        self.C_STATE = [np.array([self.C_S, self.C_L, self.C_T])]

        # 状态转移过程中需要的参数
        self.ACTION: Union[int,float] = 0
        self.ACC: float = 0  # 当前合加速度
        self.T_ACC: float = 0  # 电机牵引加速度
        self.B_ACC: float = 0  # 电机制动加速度
        self.G_ACC: float = 0  # 坡度加速度
        self.CU_ACC: float = 0  # 曲率加速度
        self.B_R_ACC: float = 0  # 基本阻力加速度
        self.ATP_LIM: float = float("inf")
        self.STA_LIM: float = float("inf")
        self.LINE = "Default"
        self.SECTION = "Default"
        self.KIN_LIST: list[float] = [0.0] * 11
        self.MOT_LIST: list[float] = [0.0] * 10
        self.ACT_TYPE: str = "C"
```

### 可能需要的函数

- [ ] _getMTF 获得当前最大牵引力
- [ ] _getMBF 获得当前最大制动力
- [ ] _getN1 获得当前牵引电机效率
- [ ] _getN1B 获得当前制动电机效率
- [ ] _getTPower 获得状态转移牵引能耗
- [ ] _getRePower 获得状态转移再生制动能量
- [ ] _getAtpLimit 计算ATP限速
- [ ] _getStaLimit 计算静态限速
- [ ] _getAction 获得当前动作
- [ ] _getGateAction 获得门控动作
- [ ] _reshapeAction 重整当前动作
- [ ] _getAcc 获得当前合加速度MBF
- [ ] _getMAcc 获得当前电机加速度
- [ ] _getTAcc 获得当前电机牵引加速度
- [ ] _getBAcc 获得当前电机制动加速度
- [ ] _getGAcc 获得当前坡度加速度
- [ ] _getRAcc 获得当前基本阻力加速度
- [ ] _getNextState 计算并进行状态转移
- [ ] _storeState 存储状态序列

---

## 环境类

*需要考虑环境是只写一个基类还是分单车多车写基类*
*环境的基类要不要具有功能需要考虑*

### 构造函数

*暂时没想好都需要什么*

| 描述 | 定义 | 数据类型 | 单位 |例子|
|:---: |:---: |  :---:  |:---:|:---:|
| 仿真列车 | SIM_TRAIN | class: Train | - | - |
| 仿真线路 | SIM_SECTION | class: Section | - | - |
| 动作空间 | action_space | spaces.Space[ActType] | - | - |
| 状态空间 | state_space | spaces.Space[ObsType] | - | - |

### 需要的函数

- [ ] reset 重置函数，env要求，重启一个episode，返回初始状态和补充说明

```python
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
```
**stable-baselines3按照total-timesteps训练，因此需要把列车信息全部重置**


- [ ] step 执行函数，env要求，进行一个timestep的仿真，返回下一状态，奖励，是否终点，是否截断，其他信息。**done**已经被弃用了，但可使用是否终点代替done。是否截断可始终置False。

```python
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
```

- [ ] render 渲染函数，env要求，目前不知道对于列车运行有什么影响，使用None或"human"配置

```python
    def render(self) -> RenderFrame | list[RenderFrame] | None:
```

### 可能需要的函数

- [ ] _getSimTrain 获得仿真列车
- [ ] _getSimLine 获得仿真线路（直接获得区间）
- [ ] _getActSpace 获得动作空间
- [ ] _getObsSpace 获得状态空间
- [ ] _processAction 预处理动作 （在env.step前先对动作进行处理，step中放动作处理函数不太合适，且step的return是有要求的）
- [ ] _comReward 计算奖励
- [ ] _comTerminal 计算是否终点
- [ ] _comInfo 计算信息没什么用，主要是给reset函数用的，应该不需要返回什么东西

**注意事项**

1. 目前这只是一个列车运行的基类，还得根据具体的场景去重写方法
2. 两条腿走路，zzc写环境，gh写基于stable-baseline3的agent，czs写自定义的agent
3. 短期目标先实现对不同reward的比较
