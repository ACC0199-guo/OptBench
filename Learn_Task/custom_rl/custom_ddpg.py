import os
import sys
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from TrainOptBench.envs.MetroRewardEnv import CustomMetroEnv
from BaseConfig import BaseConfig
from rl_model.ddpg.ddpg import DDPGAgent, ReplayBuffer
import torch
from datetime import datetime


# ==== 主训练循环 ====
def main():
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用，当前设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA不可用，使用CPU进行训练")
    # 1. 生成保存文件夹
    algo_name = "DDPG"
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(__file__), "results")
    algo_dir = os.path.join(base_dir, algo_name)
    time_dir = os.path.join(algo_dir, now)
    os.makedirs(time_dir, exist_ok=True)
    # 1. 加载default_config.yaml
    yaml_path = os.path.join(os.path.dirname(__file__), "../default_config.yaml")
    yaml_path = os.path.abspath(yaml_path)

    # 2. 用BaseConfig生成config
    base_config = BaseConfig()
    base_config.load_from_file(yaml_path)
    base_config.dispatch_para()

    env = CustomMetroEnv(base_config, reward_type="A")

    # 添加设备选择，使用自动选择的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(base_config.state_dim, base_config.action_dim, base_config.max_action, device=device)
    replay_buffer = ReplayBuffer()
    batch_size = 64

    best_reward = float('-inf')  # 新增：记录最高奖励
    
    # 初始化奖励记录列表
    episode_rewards = []
    ema_rewards = []
    alpha = 0.2  # 指数平滑因子
    ema_reward = None

    for episode in range(base_config.max_episodes):
        state, info = env.reset()
        episode_reward = 0
        for t in range(base_config.max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, float(done))
            agent.update(replay_buffer, batch_size)
            state = next_state
            episode_reward += reward
            if t % 20 == 0:
                print(f"Episode {episode}, Step {t}, State: {state}")
            if done:
                break
        print(f"Episode {episode} finished, total reward: {episode_reward}")

        # 新增：保存最高奖励模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_path = os.path.join(time_dir, "best_model.pth")
            agent.save(model_path)  # 假设DDPGAgent有save方法
            print(f"新最高奖励: {best_reward}，模型已保存到 {model_path}")
        
        # 记录当前episode奖励
        episode_rewards.append(episode_reward)
        
        # 计算指数平均奖励
        if ema_reward is None:
            ema_reward = episode_reward
        else:
            ema_reward = alpha * episode_reward + (1 - alpha) * ema_reward
        ema_rewards.append(ema_reward)


    rewards_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'ema_reward': ema_rewards
    })
    rewards_path = os.path.join(time_dir, 'training_rewards.csv')
    rewards_df.to_csv(rewards_path, index=False)
    print(f"训练奖励数据已保存到 {rewards_path}")

    # ====== 新增测试代码 ======

    # 加载最佳模型
    model_path = os.path.join(time_dir, "best_model.pth")
    agent.load(model_path)
    print(f"开始模型测试，共测试{min(base_config.eval_episodes, 10)}次...")

    # 创建Excel写入器
    excel_path = os.path.join(time_dir, "test_results.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    for test_idx in range(min(base_config.eval_episodes, 10)):
        # 重置环境进行测试
        state, info = env.reset()
        episode_reward = 0

        # 测试循环（无探索噪声）
        for t in range(base_config.max_steps):
            action = agent.select_action(state, noise_scale=0.0)  # 关闭噪声
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"测试 {test_idx + 1} 完成，奖励: {episode_reward}")

        # 提取sim_train数据并保存到DataFrame
        data = {
            'TRA_POWER_LIST': env.sim_train.TRA_POWER_LIST,
            'RE_POWER_LIST': env.sim_train.RE_POWER_LIST,
            'S_LIST': env.sim_train.S_LIST,
            'L_LIST': env.sim_train.L_LIST,
            'T_LIST': env.sim_train.T_LIST
        }

        # 添加长度检查和标准化
        lengths = [len(v) for v in data.values()]
        max_len = max(lengths)
        # 填充较短的列表以匹配最长列表的长度
        for key in data:
            if len(data[key]) < max_len:
                # 使用NaN填充缺失值
                data[key] = data[key] + [np.nan] * (max_len - len(data[key]))

        df = pd.DataFrame(data)

        # 写入Excel不同Sheet
        df.to_excel(writer, sheet_name=f'Test_{test_idx + 1}', index=False)

    # 保存Excel文件
    writer.close()
    print(f"所有测试数据已保存到 {excel_path}")
    # =========================


if __name__ == "__main__":
    main()
