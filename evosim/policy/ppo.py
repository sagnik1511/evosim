from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn import functional as F

from evosim.policy.base_policy import BasePolicy

MapState = Dict[str, List[List[int]]]
TT = torch.Tensor


class Memory:
    """Memory Class to store the game progress information"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []

    def add(
        self,
        obs: MapState,
        action: int,
        log_probs: torch.Tensor,
        reward: float,
        value: float,
    ):
        """Update current game state to memory

        Args:
            obs (MapState): Current Map State
            action (int): Current Action Taken
            log_probs (torch.Tensor): Log Probability of the taken action
            reward (float): Reward recieved through the step taken
        """
        self.states.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.values.append(value)

    def return_tensor(self) -> Tuple[TT, TT, TT, TT, TT]:
        """Return memory as tensor data

        Returns:
            Tuple[TT, TT, TT, TT]: Memory Tensors
        """
        states = torch.tensor(self.states, dtype=torch.float32)

        # Expanding dimension for proper data formatting
        actions = torch.tensor(self.actions, dtype=torch.float32).unsqueeze(1)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32).unsqueeze(1)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).unsqueeze(1)
        values = torch.tensor(self.values, dtype=torch.float32).unsqueeze(1)

        return states, actions, log_probs, rewards, values

    def clear(self):
        """Clears the Memory Buffer"""
        self.__init__()


class PPOActorCritic(nn.Module):
    """Actor-Critic neural Network Model to optimize The PPO Policy"""

    def __init__(
        self, env_side_length: int = 32, in_channels: int = 3, num_actions: int = 4
    ):
        super(PPOActorCritic, self).__init__()

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

    def forward(self, state: TT) -> Tuple[TT]:
        """Generated policy logits and value function

        Args:
            state (TT): Current Observation State

        Returns:
            Tuple[TT]: Policy Logits and value Function
        """
        x = self.fc(self.feature_extractor(state))
        policy_logits = self.policy_layer(x)
        value = self.value_layer(x)

        return policy_logits, value


class PPO(BasePolicy):
    """Proximal Policy Optimization Policy"""

    def __init__(
        self,
        actor_critic: Union[PPOActorCritic, None] = None,
        env_side_length: int = 32,
        env_channels: int = 3,
        env_actions: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.95,
        eps_clip: float = 0.2,
        k_epochs: int = 100,
        entropy_bonus: float = 0.05,
        value_scaling_factor: float = 0.5,
        learn_counter: int = 50,
    ):
        super().__init__()
        self.actor_crtitic = (
            actor_critic
            if actor_critic
            else PPOActorCritic(env_side_length, env_channels, env_actions)
        )
        self.optimizer = optim.Adam(self.actor_crtitic.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.ent_bonus = entropy_bonus
        self.vs_factor = value_scaling_factor
        self.memory = Memory()
        self.obs_counter = 0
        self.learn_counter = learn_counter

    def act(self, state: np.ndarray) -> Tuple[TT, TT, TT]:
        """Take Action over the given state

        Args:
            state (np.ndarray): Given State

        Returns:
            Tuple[TT, TT, TT]: Action, the Log Probability and Value of the action
        """

        # Making state as a batch to feed into the nn model
        state = torch.from_numpy(state).unsqueeze(0)

        # fetching class probabilities
        logits, value = self.actor_crtitic(state)

        # Fetching action from logits
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        log_probs = policy_dist.log_prob(action)

        return action, log_probs, value

    def _fetch_gae(self, rewards: TT, values: TT) -> TT:

        rewards = rewards.squeeze(1).detach().numpy().tolist()
        values = values.squeeze(1).detach().numpy().tolist()

        next_value = values[-1]

        advantages = []
        advantage = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            td_error = r + self.gamma * next_value - v
            advantage = td_error + self.gamma * advantage
            advantages.insert(0, advantage)
            next_value = v

        return torch.tensor(advantages).unsqueeze(0)

    def learn(self) -> None:
        """Learning Step"""

        # Fetch the past responses from the environment
        states, actions, stale_log_probs, rewards, values = self.memory.return_tensor()
        stale_log_probs = stale_log_probs.detach()
        returns = rewards + values
        advantages = self._fetch_gae(rewards, values)

        # Training for K_EPOCHS
        for epoch in range(self.k_epochs):
            # Zeroing the gradients of the optimizer
            self.optimizer.zero_grad()

            # Fetching the actions upon past states
            logits, values = self.actor_crtitic(states)

            # Calculate log prob and entropy for loss calculation
            logits_dist = Categorical(logits=logits)
            new_log_probs = logits_dist.log_prob(actions)
            entropy = logits_dist.entropy().mean()

            # Calculate objective fn ratio
            ratios = torch.exp(new_log_probs - stale_log_probs)

            # Calculating surrogate fns
            cpi_surr = ratios * advantages
            clip_surr = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Complete Loss formula
            ppo_loss: TT = (
                -torch.min(cpi_surr, clip_surr).mean()
                + F.mse_loss(values, returns) * self.vs_factor
                - entropy * self.ent_bonus
            )

            # Update gradients and optimize the weights
            ppo_loss.backward()
            self.optimizer.step()

    def observe(
        self, state: MapState, action: int, log_probs, reward: int, value: float
    ) -> None:
        """Load memory and learn if needed

        Args:
            state (MapState): Current Observation State
            action (int): Current Action
            log_probs (_type_): Action Log Probability
            reward (int): Reward of the Current Step
        """

        # Update counters and memory buffers
        self.obs_counter += 1
        self.memory.add(state, action, log_probs, reward, value)

        if self.obs_counter % self.learn_counter == self.learn_counter - 1:

            # Learning Phase of the Algo
            self.learn()

            # Clearing current state info
            self.memory.clear()
