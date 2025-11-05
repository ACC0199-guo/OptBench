import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from TrainOptBench.envs.MetroRewardEnv import CustomMetroEnv
from BaseConfig import BaseConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


# 定义MPC控制器类（保持不变）
class MPCController:
    def __init__(self, state_dim, action_dim, horizon=5, dt=0.1):
        self.sim_train = None
        self.constraints = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon  # MPC预测时域长度
        self.dt = dt  # 时间间隔参数

    def get_sim_train(self, sim_train):
        self.sim_train = sim_train

    def ineq_constraint(self, u_sequence, x0):
        # 将u_sequence转换为正确的形状
        u_sequence = u_sequence.reshape(-1, self.action_dim)

        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().detach().numpy()
        # 初始化状态
        x = x0.copy()
        # 获取速度限制
        speed_limit = min(self.sim_train.STA_LIM / 3.6, self.sim_train.ATP_LIM / 3.6)

        # 存储所有速度约束的违反情况
        constraints = []

        # 检查初始状态速度
        constraints.append(speed_limit - x[0])

        # 计算状态轨迹并检查每个时间步的速度
        for u in u_sequence:
            # 当前状态分解
            velocity = x[0]  # 速度
            position = x[1]  # 位置
            time = x[2]  # 时间
            acceleration = u[0]  # 控制量（加速度）

            # 状态转移方程
            new_velocity = velocity + acceleration * self.dt
            new_position = position + velocity * self.dt + 0.5 * acceleration * (self.dt ** 2)
            new_time = time + self.dt

            # 速度不能为负
            if new_velocity <= 0:
                new_velocity = 0
            if new_position <= 0:
                new_position = 0

            # 更新状态向量
            x = np.array([new_velocity, new_position, new_time])

            # 检查速度约束
            constraints.append(speed_limit - x[0])

        # 返回所有约束的数组
        return np.array(constraints)

    def set_constraints(self, x0):
        self.constraints = [
            {'type': 'ineq', 'fun': lambda u_sequence: self.ineq_constraint(u_sequence, x0)}
        ]

    def cost_function(self, u_sequence, x0, weights):
        # 根据权重矩阵构建成本函数
        # weights: [状态跟踪权重, 控制输入权重, 终端成本权重]
        Q = np.diag(weights[:self.state_dim])
        R = np.diag(weights[self.state_dim:self.state_dim + self.action_dim])
        Qf = np.diag(weights[self.state_dim + self.action_dim:])

        # 添加Tensor转NumPy数组的判断
        if isinstance(x0, torch.Tensor):
            x0 = x0.cpu().detach().numpy()
        x = x0
        cost = 0
        x_ref = np.array([self.sim_train.GATE_SPEED / 3.6, self.sim_train.GATE_LOCATION, 0])

        # 阶段成本（使用新的状态转移方程）
        for u in u_sequence.reshape(-1, self.action_dim):
            # 当前状态分解
            velocity = x[0]  # 速度
            position = x[1]  # 位置
            time = x[2]  # 时间
            acceleration = u[0]  # 控制量（加速度）

            # 累积当前状态的成本（修正：移到状态更新前）
            # 修改为x和x_ref之间的差
            x_ref[2] = time
            x_diff = x - x_ref
            cost += x_diff.T @ Q @ x_diff + u.T @ R @ u

            # 状态转移方程实现（现在在成本累积之后）
            new_velocity = velocity + acceleration * self.dt
            new_position = position + velocity * self.dt + 0.5 * acceleration * (self.dt ** 2)
            new_time = time + self.dt
            if new_velocity <= 0:
                new_velocity = 0
            if new_position <= 0:
                new_position = 0

            # 更新状态向量
            x = np.array([new_velocity, new_position, new_time])

        # 终端成本
        x_diff_terminal = x - x_ref
        cost += x_diff_terminal.T @ Qf @ x_diff_terminal
        return cost

    def solve(self, x0, weights):
        # 求解MPC优化问题获得动作序列
        u0 = np.zeros(self.horizon * self.action_dim)  # 初始猜测
        bounds = [(-1, 1) for _ in range(self.horizon * self.action_dim)]  # 动作范围限制

        # 设置约束
        self.set_constraints(x0)

        # 调用优化器求解
        result = minimize(
            self.cost_function,
            u0,
            args=(x0, weights),
            bounds=bounds,
            method='SLSQP',
            constraints=self.constraints,
            options={'maxiter': 50, 'ftol': 1e-4}
        )

        # 返回第一个控制动作
        if result.success:
            return result.x.reshape(-1, self.action_dim)[0]
        else:
            print(f"MPC优化失败: {result.message}")
            return np.zeros(self.action_dim)  # 失败时返回零动作


# 定义输出权重矩阵的Actor网络（保持不变）
class Actor(nn.Module):
    def __init__(self, state_dim, weight_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, weight_dim)
        self.activation = nn.Tanh()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        weights = self.activation(self.fc3(x))  # 输出范围[-1, 1]
        weights = F.softplus(weights) + 1e-6  # 转换为正数权重
        return weights


# 定义Critic网络（基于DDPG的Critic）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))


# 经验回放区
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(len(self.buffer), size=batch_size)
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


# 基于DDPG的MPC控制器代理
class MPC_DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, device, mpc_horizon=5, dt=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        # 权重维度: 状态跟踪权重 + 控制输入权重 + 终端成本权重
        self.weight_dim = state_dim + action_dim + state_dim

        # 初始化MPC控制器（添加dt参数）
        self.mpc_controller = MPCController(
            state_dim, action_dim,
            horizon=mpc_horizon,
            dt=dt  # 传递时间间隔
        )

        # 定义网络
        self.actor = Actor(state_dim, self.weight_dim).to(device)
        self.actor_target = Actor(state_dim, self.weight_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # 经验回放区
        self.replay_buffer = ReplayBuffer()

        # 超参数
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256

    def select_action(self, state, deterministic=False, return_tensor=False):
        # 根据当前状态获取MPC权重并求解动作
        state = torch.FloatTensor(state).to(self.device)
        weights = self.actor(state)
        # 训练时添加[0,1]随机噪声
        if not deterministic:
            noise = torch.rand_like(weights)  # 生成与weights同形状的[0,1)均匀分布噪声
            weights = weights + 0.1 * noise
        weights = weights.cpu().detach().numpy().flatten()
        action = self.mpc_controller.solve(x0=state, weights=weights)
        if return_tensor:
            return torch.FloatTensor(action).to(self.device)
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区采样
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # 转换为张量
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # 计算目标Q值
        with torch.no_grad():
            # For training (with gradient tracking)
            next_action = torch.stack([self.select_action(s.cpu().numpy(), return_tensor=True) for s in next_state]).to(self.device)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q

        # 当前Q值
        current_q = self.critic(state, action)

        # 计算Critic损失
        critic_loss = F.mse_loss(current_q, target_q)

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算Actor损失
        weights = self.actor(state)
        temp = [self.mpc_controller.solve(s.cpu().detach().numpy(), w.cpu().detach().numpy())
                for s, w in zip(state, weights)]
        temp_np = np.array(temp)  # 合并为单一numpy数组
        actions = torch.from_numpy(temp_np).float().to(self.device)  # 更高效的转换方式

        actor_loss = -self.critic(state, actions).mean()

        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store(self, transition):
        self.replay_buffer.store(transition)

    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_state_dict'])


def main():
    # 配置
    yaml_path = os.path.join(os.path.dirname(__file__), "../default_config.yaml")
    yaml_path = os.path.abspath(yaml_path)
    base_config = BaseConfig()
    base_config.load_from_file(yaml_path)
    base_config.dispatch_para()
    env = CustomMetroEnv(base_config, reward_type="A")

    # 添加设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = MPC_DDPGAgent(
        state_dim=base_config.state_dim,
        action_dim=base_config.action_dim,
        max_action=base_config.max_action,
        device=device,
        mpc_horizon=5,
        dt=1
    )

    # 创建保存文件夹
    algo_name = "MPC_DDPG"
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(__file__), "results")
    algo_dir = os.path.join(base_dir, algo_name)
    time_dir = os.path.join(algo_dir, now)
    os.makedirs(time_dir, exist_ok=True)

    best_reward = float('-inf')  # 记录最高奖励
    episode_rewards = []
    ema_rewards = []
    alpha = 0.2  # 指数平滑因子
    ema_reward = None

    # 训练循环
    for episode in range(base_config.max_episodes):
        obs, info = env.reset()
        episode_reward = 0
        agent.mpc_controller.get_sim_train(env.sim_train)

        for t in range(base_config.max_steps):
            env.sim_train._getGateData()
            env.sim_train._getLimit()
            action = agent.select_action(obs,deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.store((obs, action, reward, next_obs, float(done)))

            # 更新网络
            if t % 40 == 0:
                agent.update()

            obs = next_obs
            episode_reward += reward

            if t % 20 == 0:
                print(f"Episode {episode}, Step {t}, State: {next_obs[:5]}...")

            if done:
                break

        print(f"Episode {episode} finished, total reward: {episode_reward}")
        print("sim_train状态：", env.sim_train.N_S, env.sim_train.N_L, env.sim_train.N_T)

        # 保存最高奖励模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_path = os.path.join(time_dir, "best_model.pth")
            agent.save(model_path)
            print(f"新最高奖励: {best_reward}，模型已保存到 {model_path}")

        # 记录奖励
        episode_rewards.append(episode_reward)

        # 计算指数平均奖励
        if ema_reward is None:
            ema_reward = episode_reward
        else:
            ema_reward = alpha * episode_reward + (1 - alpha) * ema_reward
        ema_rewards.append(ema_reward)

    # 保存奖励数据
    rewards_df = pd.DataFrame({
        'episode': range(len(episode_rewards)),
        'reward': episode_rewards,
        'ema_reward': ema_rewards
    })
    rewards_path = os.path.join(time_dir, 'training_rewards.csv')
    rewards_df.to_csv(rewards_path, index=False)
    print(f"训练奖励数据已保存到 {rewards_path}")

    # 测试代码
    model_path = os.path.join(time_dir, "best_model.pth")
    agent.load(model_path)
    print(f"开始模型测试，共测试{min(base_config.eval_episodes, 10)}次...")

    excel_path = os.path.join(time_dir, "test_results.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')

    for test_idx in range(min(base_config.eval_episodes, 10)):
        obs, info = env.reset()
        episode_reward = 0

        for t in range(base_config.max_steps):
            env.sim_train._getGateData()
            env.sim_train._getLimit()
            action = agent.select_action(obs,deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            if done:
                break

        print(f"测试 {test_idx + 1} 完成，奖励: {episode_reward}")

        # 保存测试结果
        data = {
            'TRA_POWER_LIST': env.sim_train.TRA_POWER_LIST,
            'RE_POWER_LIST': env.sim_train.RE_POWER_LIST,
            'S_LIST': env.sim_train.S_LIST,
            'L_LIST': env.sim_train.L_LIST,
            'T_LIST': env.sim_train.T_LIST
        }

        # 处理数据长度不一致问题
        lengths = [len(v) for v in data.values()]
        max_len = max(lengths)
        for key in data:
            if len(data[key]) < max_len:
                data[key] = data[key] + [np.nan] * (max_len - len(data[key]))

        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=f'Test_{test_idx + 1}', index=False)

    writer.close()
    print(f"所有测试数据已保存到 {excel_path}")


if __name__ == "__main__":
    main()

