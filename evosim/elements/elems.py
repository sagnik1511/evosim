from dataclasses import dataclass
from abc import ABC, abstractmethod
from evosim.maps.cells import Pos, Cell


@dataclass
class Element(ABC):

    def __init__(self, pos: Pos):
        self.pos: Pos = pos
        self.static: bool = False


class Resource(Element):

    def __init__(self, pos: Pos, hp: int):
        super().__init__(pos=pos)
        self.hp: int = hp
        self.max_hp: int = 1000
        self.grow_delta: int = 1

    def grow(self) -> None:
        self.hp = max(self.hp, self.hp + self.grow_delta)


class Obstacle(Element):

    def __init__(self, pos: Pos):
        super().__init__(pos=pos)
        self.static = True

    @abstractmethod
    def detoriate(self):
        pass


# if __name__ == "__main__":

#     res = Resources(Pos(2,3))
#     print(res.pos)
