import os
import yaml
from typing import Union


class BaseConfig:
    def __init__(self):
        self.config_data = {}

        ########## Train Config ########
        self.weight: float = 0.0
        self.d_a: float = 0.0
        self.d_b: float = 0.0
        self.d_c: float = 0.0
        self.n2: float = 0.0
        self.n3: float = 0.0
        self.n4: float = 0.0
        self.mta: float = 0.0
        self.mba: float = 0.0
        self.kin_list: list[float] = [0.0] * 11
        self.mot_list: list[float] = [0.0] * 10
        self.type: str = "C"
        self.train_model: str = "Default"
        self.max_speed: float = 0.0
        self.err_speed: float = 0.0  # 电制动速度误差系数，由于低速运行时制动为空气制动，在该仿真中全部认为是电制动，因此制动系数存在非0区段，使用该误差系数对制动系数曲线进行平移

        ######## Line Config  ##########
        self.line: str = "Default"
        self.section_name: str = "Default"

        ######## Training Config ######
        self.max_episodes: int = 0
        self.max_steps: int = 0
        self.max_sb3steps: int = 0
        self.state_dim: int = 0
        self.action_dim: int = 0
        self.max_action: int = 0
        self.eval_episodes: int = 0

    def load_from_file(self, file_path: str):
        """
        从YAML文件加载配置

        Args:
            file_path: YAML文件路径
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.config_data = yaml.safe_load(f)
            if self.config_data:
                self.config_data.update(self.config_data)

    def dispatch_para(self):
        ########   Train Para   ##########
        self.weight = self.config_data["Train"]["weight"]
        self.d_a = self.config_data["Train"]["d_a"]
        self.d_b = self.config_data["Train"]["d_b"]
        self.d_c = self.config_data["Train"]["d_c"]
        self.n2 = self.config_data["Train"]["n2"]
        self.n3 = self.config_data["Train"]["n3"]
        self.n4 = self.config_data["Train"]["n4"]
        self.mta = self.config_data["Train"]["mta"]
        self.mba = self.config_data["Train"]["mba"]
        self.kin_list = self.config_data["Train"]["kin_list"]
        self.mot_list = self.config_data["Train"]["mot_list"]
        self.type = self.config_data["Train"]["type"]
        self.train_model = self.config_data["Train"]["train_model"]
        self.max_speed = self.config_data["Train"]["max_speed"]
        self.err_speed = self.config_data["Train"]["err_speed"]

        ######## Line Para ############
        self.line = self.config_data["Line"]["line"]
        self.section_name = self.config_data["Line"]["section"]

        ####### Training Para  ############
        self.max_episodes = self.config_data["Training"]["max_episodes"]
        self.max_steps = self.config_data["Training"]["max_steps"]
        self.max_sb3steps = self.config_data["Training"]["max_sb3steps"]
        self.state_dim = self.config_data["Training"]["state_dim"]
        self.action_dim = self.config_data["Training"]["action_dim"]
        self.max_action = self.config_data["Training"]["max_action"]
        self.eval_episodes = self.config_data["Training"]["eval_episodes"]

