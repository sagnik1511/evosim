from abc import ABC, abstractmethod
from typing import Dict, List

MapState = Dict[str, List[List[int]]]


class BasePolicy(ABC):

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def act(self, state: MapState):
        pass
