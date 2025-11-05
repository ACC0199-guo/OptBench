import numpy as np
import torch
import torch.optim as optim
from rl_model.ppo.actor_critic import Actor, Critic
import os
from TrainOptBench.envs.MetroRewardEnv import CustomMetroEnv
from BaseConfig import BaseConfig

# 加载线路
from TrainOptBench.lines.metro_lines.ChengDu17 import Section


def ppo_update(actor, critic, optimizer_actor, optimizer_critic, states, actions, log_probs_old, returns, advantages, clip_epsilon=0.2, epochs=10, batch_size=64):
    dataset = torch.utils.data.TensorDataset(states, actions, log_probs_old, returns, advantages)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for batch in loader:
            b_states, b_actions, b_log_probs_old, b_returns, b_advantages = batch
            mu, std = actor(b_states)
            dist = torch.distributions.Normal(mu, std)
            log_probs = dist.log_prob(b_actions).sum(axis=-1, keepdim=True)
            ratio = (log_probs - b_log_probs_old).exp()
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            value = critic(b_states)
            critic_loss = (b_returns - value).pow(2).mean()

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()


def collect_trajectory(env, actor, critic, max_steps, gamma=0.99, lam=0.95, print_freq=10):
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    obs, info = env.reset()
    for t in range(max_steps):
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, std = actor(state)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            value = critic(state)
        action_np = action.squeeze(0).numpy()
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        # 记录
        states.append(obs)
        actions.append(action_np)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.item())
        values.append(value.item())

        obs = next_obs

        if t % print_freq == 0:
            print(f"Step {t}: state={obs}")

        if done:
            break

    # 计算GAE优势
    returns, advantages = [], []
    gae = 0
    next_value = 0 if dones[-1] else critic(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
        returns.insert(0, gae + values[i])
    # 转为tensor
    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        torch.tensor(log_probs, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(returns, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)
    )


# 1. 加载default_config.yaml
yaml_path = os.path.join(os.path.dirname(__file__), "../default_config.yaml")
yaml_path = os.path.abspath(yaml_path)

# 2. 用BaseConfig生成config
base_config = BaseConfig()
base_config.load_from_file(yaml_path)
base_config.dispatch_para()

# 环境参数
env = CustomMetroEnv(base_config, reward_type="A")
state_dim = 3
action_dim = 1

# 网络和优化器
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
optimizer_actor = optim.Adam(actor.parameters(), lr=3e-4)
optimizer_critic = optim.Adam(critic.parameters(), lr=1e-3)

# 训练主循环
num_episodes = 1000
max_steps_per_episode = 200
for episode in range(num_episodes):
    states, actions, log_probs_old, returns, advantages = collect_trajectory(
        env, actor, critic, max_steps=max_steps_per_episode, print_freq=20
    )
    ppo_update(actor, critic, optimizer_actor, optimizer_critic, states, actions, log_probs_old, returns, advantages)
    print(f"Episode {episode} finished, total steps: {len(states)}")
