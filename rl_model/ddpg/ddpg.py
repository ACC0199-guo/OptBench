import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 动态添加项目根目录到sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ==== 网络结构 ====
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.net(x) * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))


# ==== 经验回放 ====
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s_),
            torch.FloatTensor(d).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# ==== DDPG Agent ====
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, actor_lr=1e-3, critic_lr=1e-3, device=None):
        # 添加设备参数，默认自动选择
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 将所有模型移动到指定设备
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state, noise_scale=0.1):
        # 输入状态移动到指定设备
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).detach().cpu().numpy()[0]
        action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -self.actor.max_action, self.actor.max_action)

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return
        s, a, r, s_, d = replay_buffer.sample(batch_size)
        # 将采样数据移动到指定设备
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        d = d.to(self.device)
        
        with torch.no_grad():
            a_next = self.actor_target(s_)
            q_next = self.critic_target(s_, a_next)
            q_target = r + self.gamma * (1 - d) * q_next
        q = self.critic(s, a)
        critic_loss = nn.MSELoss()(q, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Polyak update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, save_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, save_path)
        print(f"模型已保存到 {save_path}")

    def load(self, load_path):
        # 加载模型时使用当前设备
        checkpoint = torch.load(load_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"模型已从 {load_path} 加载")
