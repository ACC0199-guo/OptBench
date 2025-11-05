import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        # 增加网络层数，从原来的2层增加到4层
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        # 为每一层添加残差连接所需的投影层
        self.residual_fc1 = nn.Linear(128, 128)
        self.residual_fc2 = nn.Linear(128, 128)
        self.residual_fc3 = nn.Linear(128, 128)
        # 输出层保持不变
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        # 第一层及残差连接
        x = F.relu(self.fc1(state))
        x1 = x
        # 第二层及残差连接
        x = self.fc2(x)
        x = x + F.relu(self.residual_fc1(x1))
        x = F.relu(x)
        x2 = x
        # 第三层及残差连接
        x = self.fc3(x)
        x = x + F.relu(self.residual_fc2(x2))
        x = F.relu(x)
        x3 = x
        # 第四层及残差连接
        x = self.fc4(x)
        x = x + F.relu(self.residual_fc3(x3))
        x = F.relu(x)
        # 输出均值和标准差
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in ind]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )


class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=3e-3, target_entropy=None, buffer_size=100000, batch_size=256, device=None):
        # 添加设备参数，默认自动选择
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 将所有模型移动到指定设备
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        # 将log_alpha移动到指定设备
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.max_action = max_action

    def select_action(self, state, deterministic=False):
        # 输入状态移动到指定设备
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.max_action
            return action.detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]

    def store(self, transition):
        self.replay_buffer.add(transition)

    def update(self):
        if len(self.replay_buffer.storage) < self.batch_size:
            return

        s, a, r, s_, d = self.replay_buffer.sample(self.batch_size)
        # 将采样数据移动到指定设备
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        d = d.to(self.device)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(s_)
            q1_next, q2_next = self.critic_target(s_, next_action)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_prob
            target_q = r + self.gamma * (1 - d) * q_next

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor更新
        new_action, log_prob = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_action)
        actor_loss = (self.log_alpha.exp() * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # alpha自动调整
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 软更新
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, save_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, save_path)
        print(f"模型已保存到 {save_path}")

    def load(self, load_path):
        # 加载模型时使用当前设备
        checkpoint = torch.load(load_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if 'critic_target' in checkpoint:
            self.critic_target.load_state_dict(checkpoint['critic_target'])
        print(f"模型已从 {load_path} 加载")
