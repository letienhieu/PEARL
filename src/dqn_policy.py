import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=20, action_dim=2):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))
    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

class DQNPolicy:
    def __init__(self, state_dim=20, action_dim=2, lr=1e-4):
        self.policy_net = DuelingDQN(state_dim, action_dim)
        self.target_net = DuelingDQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.gamma = 0.99

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(2)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state_t).argmax().item()

    def update(self, batch_size=64):
        if len(self.memory) < batch_size: return None
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_q = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # Double DQN (van Hasselt et al. 2015): policy_net selects, target_net evaluates
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            expected_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(curr_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
