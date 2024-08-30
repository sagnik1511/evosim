import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import torch

from evosim.maps.base_map import BaseMap
from evosim.maps.cells import Pos
from evosim.policy.base_policy import BasePolicy
from evosim.utils.logger import get_logger

logger = get_logger()

MapState = Dict[str, List[List[int]]]


class Agent(ABC):

    def __init__(self, policy: Union[None, BasePolicy]):
        self.pos = Pos(0, 0)
        self.policy = policy
        self.save_directory = "agents"
        self.artifact_name = self.__class__.__name__

    @abstractmethod
    def observe(self, env: BaseMap):
        pass

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def save(self):
        """Save the agent in a pkl file"""
        if not os.path.isdir(self.save_directory):

            logger.info(f"Creating agent directory at - {self.save_directory}")
            os.makedirs(self.save_directory)

        artifact_path = os.path.join(self.save_directory, self.artifact_name) + ".pkl"

        with open(artifact_path, "w") as f:
            pickle.dump(self, f)

        logger.info(f"Agent has been saved at - {artifact_path}")

    def load(self, artifact_path) -> "BasePolicy":
        """Loads the saved agent

        Args:
            artifact_path (str): Path of the checkpoint

        Raises:
            FileNotFoundError: If the checkpoint file is missing

        Returns:
            BasePolicy: The current state of the policy
        """
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(f"{artifact_path} doesn't exist")

        with open(artifact_path, "r") as f:
            instance = pickle.load(f)

        return instance


class L1Agent(Agent):

    def __init__(self, policy: Union[None, BasePolicy]):
        super().__init__(policy=policy)
        self.hp = 1000
        self.run_delta = 10

    def act(self, state: MapState):
        obs = np.stack([state["Rock"], state["Wood"], state["Agent"]], axis=0)
        probs = self.policy.act(obs)
        return torch.argmax(probs, axis=1).item(), probs.detach().numpy()

    def observe(self, obs, action, log_probs, reward):
        obs = np.stack([obs["Rock"], obs["Wood"], obs["Agent"]], axis=0)
        self.policy.observe(obs, action, log_probs, reward)

    def __str__(self):
        return f"L1Agent({self.pos})"

    def run(self):
        self.hp -= self.run_delta
