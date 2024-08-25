from abc import ABC, abstractmethod
from evosim.maps.base_map import BaseMap
from evosim.maps.cells import Pos


class Agent(ABC):

    def __init__(self, pos: Pos):
        self.pos = pos

    @abstractmethod
    def _observe(self, env: BaseMap):
        pass

    @abstractmethod
    def _act(self):
        pass

    @abstractmethod
    def run(self):
        pass


class L1Agent(Agent):

    def __init__(self, pos: Pos):
        super().__init__(pos=pos)
        self.hp = 100
        self.run_delta = 10

    def _act(self):
        pass

    def _observe(self):
        pass

    def __str__(self):
        return f"L1Agent({self.pos})"

    def run(self):
        self.hp -= self.run_delta
