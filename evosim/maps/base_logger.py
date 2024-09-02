from abc import ABC, abstractmethod
from typing import Dict, List

MapState = Dict[str, List[List[int]]]


class BaseLogger(ABC):

    def __init__(self, project_name):
        self.project_name = project_name

    @abstractmethod
    def log_step(self, episode: int, obs: MapState, *args) -> None:
        """Logs each step in WandB

        Args:
            episode (int): Number of the Episode
            obs (MapState): Step observation
        """
        pass

    @abstractmethod
    def log_episode(self, episode: int, *args) -> None:
        """Logs each episode in WandB

        Args:
            episode (int): Number of the Episode
        """
        pass
