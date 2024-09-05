import random
from collections import deque
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from evosim.policy.base_policy import BasePolicy

MapState = Dict[str, List[List[int]]]
TT = torch.Tensor


class ActorCriticNetwork(nn.Module):

    def __init__(
        self, env_side_length: int = 32, in_channels: int = 3, num_actions: int = 4
    ):
        super(ActorCriticNetwork, self).__init__()
        self.input_dim = env_side_length
        self.num_action = num_actions
        self.hidden_dim = 32

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear((self.hidden_dim // 2) * (self.input_dim // 4) ** 2, 256)
        self.policy_layer = nn.Linear(256, num_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, states: TT) -> TT:
        return self.softmax(self.policy_layer(self.fc(self.feature_extractor(states))))


class Memory:

    def __init__(self, max_mem_size: int = 10000, batch_size: int = 100):
        self.buffer = deque(maxlen=max_mem_size)
        self.bs = batch_size

    def add(self, state, action, reward, state_, done):
        self.buffer.append([state, action, reward, state_, done])

    def sample(self):
        batch = random.sample(self.buffer, self.bs)
        states, actions, rewards, states_, dones = map(np.stack, zip(*batch))

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.long).unsqueeze(1)

        return states, actions, rewards, states_, dones

    def __len__(self):
        return len(self.buffer)


class DDQN(BasePolicy):

    def __init__(
        self,
        env_side_length: int = 32,
        env_channels: int = 3,
        env_actions: int = 4,
        gamma: float = 0.9,
        eps: float = 0.999,
        eps_decay: float = 0.999,
        eps_min: float = 0.1,
        lr: float = 3e-4,
        buffer_size: int = 100000,
        batch_size: int = 100,
        learn_counter: int = 100,
    ):
        self.online_net = ActorCriticNetwork(env_side_length, env_channels, env_actions)
        self.target_net = ActorCriticNetwork(env_side_length, env_channels, env_actions)
        self.act_n = env_actions
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = Memory(buffer_size, batch_size)
        self.obs_counter = 0
        self.learn_counter = learn_counter

    def act(self, state: np.ndarray) -> int:

        # Update Epsilon (Epsilo Annealing)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        act_prob = random.random()

        if act_prob < self.eps:

            # Chossing Random Action
            action = random.randint(0, self.act_n - 1)

            return action
        else:

            # Fetch result from Online Network
            state = torch.tensor(state).unsqueeze(0)

            self.online_net.eval()
            with torch.no_grad():
                # Generating Net Predictions
                preds = self.online_net(state)
                action = torch.argmax(preds).squeeze(0)
            return action.item()

    def learn(self):

        # Make gradients 0 at Optimizer
        self.optimizer.zero_grad()

        # Load previous runs from replay buffer
        states, actions, rewards, states_, dones = self.memory.sample()

        # Predicting actions from Online Network
        online_preds = self.online_net(states)
        q_values = online_preds.gather(1, actions)

        # Prediction over next state
        next_state_preds = self.target_net(states_)
        next_max_q_values = next_state_preds.detach().max(1)[0].unsqueeze(1)

        # Generating next q values
        next_q_values = rewards + self.gamma * next_max_q_values * (1 - dones)

        # Defining loss function
        loss = F.mse_loss(q_values, next_q_values)

        # Update gradients and optimize the weights
        loss.backward()
        self.optimizer.step()

    def observe(self, state, action, reward, state_, done):

        # Update counters and memory buffers
        self.obs_counter += 1
        self.memory.add(state, action, reward, state_, done)

        if self.obs_counter % self.learn_counter == self.learn_counter - 1:
            # Learning Phase of the Algo
            self.learn()
