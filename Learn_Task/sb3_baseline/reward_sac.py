import os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from TrainOptBench.envs.MetroRewardEnv import CustomMetroEnv
from BaseConfig import BaseConfig
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList


class PrintSimTrainCallback(BaseCallback):
    def __init__(self, env, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        # 每print_freq步打印一次
        if self.n_calls % self.print_freq == 0:
            print(f"Step {self.n_calls}: sim_train状态: {self.env.sim_train.C_S, self.env.sim_train.C_L, self.env.sim_train.C_T}")
        return True

    def _on_rollout_end(self):
        # 每个rollout结束时也可以打印
        print(f"Rollout结束: sim_train状态: {self.env.sim_train.C_S, self.env.sim_train.C_L, self.env.sim_train.C_T}")


def main():
        # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用，当前设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA不可用，使用CPU进行训练")
    # 1. 生成保存文件夹
    algo_name = "SAC"
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.getcwd(), "results")
    algo_dir = os.path.join(base_dir, algo_name)
    time_dir = os.path.join(algo_dir, now)
    os.makedirs(time_dir, exist_ok=True)

    # 2. 环境配置
    yaml_path = os.path.join(os.path.dirname(__file__), "../default_config.yaml")
    yaml_path = os.path.abspath(yaml_path)
    base_config = BaseConfig()
    base_config.load_from_file(yaml_path)
    base_config.dispatch_para()
    env = CustomMetroEnv(base_config, reward_type="A")
    env.reset()

    # 3. 创建模型
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        device="cuda",
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto"
    )

    # 4. 创建EvalCallback，自动保存最高奖励模型
    eval_env = CustomMetroEnv(base_config, reward_type="A")  # 用于评估
    eval_env.reset()

    # 自定义打印callback
    print_callback = PrintSimTrainCallback(env, print_freq=1000)

    # EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=time_dir,
        log_path=time_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # 组合callback
    callback = CallbackList([print_callback, eval_callback])

    # 5. 训练
    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    print(f"训练结束，最优模型已保存在：{time_dir}")


if __name__ == "__main__":
    main()
