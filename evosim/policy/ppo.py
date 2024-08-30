from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from evosim.policy.base_policy import BasePolicy

MapState = Dict[str, List[List[int]]]


class ActorCritic(nn.Module):

    def __init__(
        self, env_side_length: int = 32, in_channels: int = 3, num_actions: int = 4
    ):
        super(ActorCritic, self).__init__()

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
        self.value_layer = nn.Linear(256, 1)

    def forward(self, states):
        x = self.feature_extractor(states)
        x = self.fc(x)
        policy_logits = self.policy_layer(x)
        value = self.value_layer(x)

        return policy_logits, value

    def act(self, state):
        policy_logits, _ = self.forward(state)
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, state, action):
        policy_logits, value = self.forward(state)
        dist = Categorical(logits=policy_logits)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_log_probs, torch.squeeze(value), dist_entropy


class Memory:
    def __init__(self):
        self.clear_memory()

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_proba = []
        self.rewards = []

    def add(self, state, action, log_prob, reward):
        self.states.append(state)
        self.actions.append(action)
        self.log_proba.append(log_prob)
        self.rewards.append(reward)


class PPO(BasePolicy):

    def __init__(
        self,
        actor_critic: Union[None, nn.Module] = None,
        lr: float = 3e-4,
        gamma: float = 0.98,
        eps_clip: float = 0.2,
        max_epochs: int = 3,
    ):
        self.actor_critic = actor_critic if actor_critic else ActorCritic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.max_epochs = max_epochs
        self.memory = Memory()
        self.learning_counter = 100
        self.counter = 0

    def learn(self):

        states = torch.tensor(self.memory.states, dtype=torch.float32)
        actions = torch.tensor(self.memory.actions, dtype=torch.float32)
        rewards = torch.tensor(self.memory.actions, dtype=torch.float32)
        stale_log_proba = torch.tensor(self.memory.log_proba, dtype=torch.float32)

        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - self.actor_critic.evaluate(states, actions)[1].detach()

        for _ in range(self.max_epochs):
            log_probs, state_values, dist_entropy = self.actor_critic.evaluate(
                states, actions
            )
            ratios = torch.exp(log_probs.unsqueeze(1) - stale_log_proba)
            surr1 = ratios * advantages.unsqueeze(1)
            surr2 = torch.clamp(
                ratios, 1 - self.eps_clip, 1 + self.eps_clip
            ) * advantages.unsqueeze(1)
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * nn.MSELoss()(state_values, returns)
                - 0.01 * dist_entropy.unsqueeze(1)
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def act(self, state: np.ndarray):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.actor_critic(state)[0]

    def observe(self, state: MapState, action, log_proba, reward):
        self.counter += 1
        self.memory.add(state, action, log_proba, reward)
        if self.counter % self.learning_counter == self.learning_counter - 1:
            self.learn()
            self.memory.clear_memory()
