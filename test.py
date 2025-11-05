from TrainOptBench.trains.BaseMetro import BaseMetro
from BaseConfig import BaseConfig
from TrainOptBench.lines.metro_lines.ChengDu17 import Section
import numpy as np
import random

config = BaseConfig()
config.load_from_file("default_config.yaml")
config.dispatch_para()
sim_train = BaseMetro(config)
sim_train.SECTION = Section[sim_train.SECTION_NAME]
for i in range(10):
    # a = np.random.rand(1)
    a = random.random()
    sim_train.execute(a)

print(sim_train.SECTION_NAME)
