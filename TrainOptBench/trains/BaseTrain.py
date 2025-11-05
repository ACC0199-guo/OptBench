import numpy as np
from typing import Union


class BaseTrain:
    """Base class for all trains."""

    def __init__(self, train_config):
        """ Initialise the train model.

        Parameters
        ----------
        train_config : yaml, load from local directory

         """

        ########     Train Parameters     ##########
        self.WEIGHT: float = train_config.weight
        self.D_A: float = train_config.d_a
        self.D_B: float = train_config.d_b
        self.D_C: float = train_config.d_c
        self.N1: float = 0.0
        self.N1_B: float = 0.0
        self.N2: float = train_config.n2
        self.N3: float = train_config.n3
        self.N4: float = train_config.n4
        self.M_T_F: float = 0.0
        self.M_B_F: float = 0.0
        self.M_T_A: float = train_config.mta
        self.M_B_A: float = train_config.mba
        self.KIN_LIST: list[float] = train_config.kin_list
        self.MOT_LIST: list[float] = train_config.mot_list
        self.MAX_SPEED: float = train_config.max_speed
        self.ERR_SPEED: float = train_config.err_speed

        ########     Train States     ##########
        self.C_S: float = 0.0
        self.C_L: float = 0.0
        self.C_T: float = 0.0
        self.B_S: float = 0.0
        self.B_L: float = 0.0
        self.B_T: float = 0.0
        self.N_S: float = 0.0
        self.N_L: float = 0.0
        self.N_T: float = 0.0
        self.C_STATE: list[np.ndarray] = [np.array([self.C_S, self.C_L, self.C_T])]
        self.ACTION: Union[int, float] = 0.0
        self.ACC: float = 0.0
        self.T_ACC: float = 0.0
        self.B_ACC: float = 0.0
        self.G_ACC: float = 0.0
        self.CU_ACC: float = 0.0
        self.B_R_ACC: float = 0.0

        ########     Env Constraints     ##########

        self.ATP_LIM: float = 0.0
        self.STA_LIM: float = 0.0
        self.LINE: str = train_config.line
        self.SECTION_NAME: str = train_config.section_name
        self.SECTION = None
        self.ACT_TYPE: str = train_config.type

        # 超速标志位，默认为False
        self.OVER_SPEED: bool = False

        ##每一个仿真step对应的列车运行时间，默认为1s
        self.TIME_STEP: float = 1.0

        # step能耗
        self.STEP_T_POWER: float = 0.0
        self.STEP_RE_POWER: float = 0.0

        #########   Store List   ##########
        self.TRA_POWER_LIST = [0.0]
        self.RE_POWER_LIST = [0.0]
        self.S_LIST = [0.0]
        self.L_LIST = [0.0]
        self.T_LIST = [0.0]

    def _getMTF(self):
        """
        Calculates the maximum traction forces (kN) with current speed.

        Must be implemented in a subclass.

        :return:The current maximum traction forces (kN)

        """

        raise NotImplementedError

    def _getMBF(self):
        """
        Calculates the maximum braking forces (kN) with current speed.
        Must be implemented in a subclass.

        :return:The current maximum braking forces (kN)
        """
        raise NotImplementedError

    def _getN1(self):
        """
        Calculates the current traction efficiency parameter of motor.
        Must be implemented in a subclass.

        :return:The current traction efficiency parameter of motor
        """
        raise NotImplementedError

    def _getN1B(self):
        """
        Calculates the current braking efficiency parameter of motor.
        Must be implemented in a subclass.

        :return:The current braking efficiency parameter of motor
        """
        raise NotImplementedError

    def _getStepPower(self):
        """
        Calculates the overall power of one simulation step.
        :return:
        """
        raise NotImplementedError

    def _getTPower(self):
        """
        Calculates the traction power of one simulation step.
        Must be implemented in a subclass.

        :return: The traction power of one simulation step
        """
        raise NotImplementedError

    def _getRePower(self):
        """
        Calculates the regenerative power of one simulation step.
        Must be implemented in a subclass.

        :return: The regenerative power of one simulation step
        """
        raise NotImplementedError

    def _getLimit(self):
        """
        Calculates the ATP and Static speed limit of next step.
        Must be implemented in a subclass.

        :return: The ATP and Static speed limit of next step
        """
        raise NotImplementedError

    def _getAtpLimit(self):
        """
        Calculates the ATP speed limit of next step.
        Must be implemented in a subclass.

        :return: The ATP speed limit of next step
        """
        raise NotImplementedError

    def _getStaLimit(self):
        """
        Calculates the static speed limit of next step.
        Must be implemented in a subclass.

        :return: The static speed limit of next step
        """
        raise NotImplementedError

    def _getAction(self, action):
        """
        Gets the original action output by the neural network.
        Must be implemented in a subclass.

        :return: The original action output by the neural network
        """
        raise NotImplementedError

    def _getGateAction(self):
        """
        Gets the potential action output by the influence of gates.
        Must be implemented in a subclass.

        :return: The potential action output by the influence of gates
        """
        raise NotImplementedError

    def _reshapeAction(self):
        """
        Combines the action output by the neural network with the potential action.
        Must be implemented in a subclass.

        :return: The reshaped action combined with original action and potential action
        """
        raise NotImplementedError

    def _getAcc(self):
        """
        Gets the current acceleration.
        Must be implemented in a subclass.

        :return: The acceleration of the train at current step
        """
        raise NotImplementedError

    def _getMAcc(self):
        """
        Gets the current motor acceleration.
        Must be implemented in a subclass.

        :return: The acceleration of the motor at current step
        """
        raise NotImplementedError

    def _getTAcc(self):
        """
        Gets the current motor traction acceleration.
        Must be implemented in a subclass.

        :return: The traction acceleration of the motor at current step
        """
        raise NotImplementedError

    def _getBAcc(self):
        """
        Gets the current motor braking acceleration.
        Must be implemented in a subclass.

        :return: The braking acceleration of the motor at current step
        """
        raise NotImplementedError

    def _getGAcc(self):
        """
        Gets the current gradient acceleration.
        Must be implemented in a subclass.

        :return: The gradient acceleration of the motor at current step
        """
        raise NotImplementedError

    def _getCuAcc(self):
        """
        Gets the current curve acceleration.
        Must be implemented in a subclass.

        :return: The gradient acceleration of the motor at current step
        """
        raise NotImplementedError

    def _getRAcc(self):
        """
        Gets the current basic resistance acceleration.
        Must be implemented in a subclass.

        :return: The basic resistance acceleration of the motor at current step
        """
        raise NotImplementedError

    def _getNextState(self):
        """
        Gets to next state.
        Must be implemented in a subclass.

        :return: The next state of the train
        """
        raise NotImplementedError

    def _trans2NextState(self):
        """
        Trans to next state.
        Must be implemented in a subclass.

        """
        raise NotImplementedError

    def _checkOverSpeed(self):
        """
        Checks if the speed is over the speed limit.

        :return:
        """
        raise NotImplementedError

    def _storeStepInfo(self):
        """
        Stores the info of each simulation step.

        :return:
        """
        raise NotImplementedError
