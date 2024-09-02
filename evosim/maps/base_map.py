from abc import abstractmethod
from typing import Any, Dict, List

from evosim.maps.cells import Cell, Pos


class BaseMap:

    def __init__(self, side_length):
        self.side_n = side_length
        self.set_map()

    def set_map(self) -> None:
        """Generates base map structure"""
        self.__struct: List[List[Cell]] = [
            [Cell(Pos(idx, idy)) for idy in range(self.side_n)]
            for idx in range(self.side_n)
        ]

    def _is_pos_free(self, pos: Pos) -> bool:
        """Checks if position is free or not

        Args:
            pos (Pos): Corresponding posiiton

        Raises:
            ValueError: If position is out of bound

        Returns:
            bool: if the position is empty
        """
        if pos.x < 0 or pos.x >= self.side_n:
            raise ValueError(f"Cell X position out of bound")
        if pos.y < 0 or pos.y >= self.side_n:
            raise ValueError(f"Cell Y position out of bound")

        return self.__struct[pos.x][pos.y]._is_free()

    def assign_element(self, element: Any, pos: Pos) -> bool:
        """Assigns element to a map struct if possible

        Args:
            element (Any): Game Element

        Returns:
            bool: Returns True is element is assigned else False
        """

        if self._is_pos_free(pos):
            # Cell is empty, assigning element to it
            self.__struct[pos.x][pos.y] = element
            return True

        return False

    def fetch_cell(self, pos: Pos) -> Cell:
        """Fetches element from respective position in the map if found

        Args:
            pos (Pos): Position in the struct

        Raises:
            ValueError: If cell is out of bound

        Returns:
            Cell: Fetches the corresponding cell
        """
        if pos.x < 0 or pos.x >= self.side_n:
            raise ValueError(f"Cell X position out of bound")
        if pos.y < 0 or pos.y >= self.side_n:
            raise ValueError(f"Cell Y position out of bound")

        return self.__struct[pos.x][pos.y]

    @abstractmethod
    def reset(self) -> Dict[str, List[List[int]]]:
        pass

    @abstractmethod
    def _get_current_state(self):
        pass

    @abstractmethod
    def log_observation(self):
        pass


# if __name__ == "__main__":
#     map = BaseMap(10)
#     pos = Pos(0, 0)
#     print(map.fetch_element(pos))
