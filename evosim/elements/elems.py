from abc import ABC, abstractmethod
from dataclasses import dataclass

from evosim.maps.cells import Pos


@dataclass
class Element(ABC):

    def __init__(self, pos: Pos):
        self.pos: Pos = pos
        self.static: bool = False


class Resource(Element):

    def __init__(self, pos: Pos, hp: int, max_hp: int, grow_delta: int = 1):
        super().__init__(pos=pos)
        self.hp = hp
        self.max_hp = max_hp
        self.grow_delta: grow_delta

    def grow(self) -> None:
        self.hp += self.grow_delta
        self.hp = min(self.max_hp, self.hp)


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
