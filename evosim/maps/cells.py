from typing import Any, Tuple


class Pos:
    def __init__(self, x, y):
        self.x: int = x
        self.y: int = y

    def __str__(self):
        return f"Pos({self.x}, {self.y})"

    def __add__(self, pos: "Pos"):
        x = self.x + pos[0]
        y = self.y + pos[1]

        return Pos(x, y)

    def __radd__(self, pos: Tuple[int, int]):
        x = self.x + pos[0]
        y = self.y + pos[1]

        return Pos(x, y)

    def __eq__(self, pos: "Pos"):
        return self.x == pos.x and self.y == pos.y


class Cell:

    def __init__(self, pos: Pos):
        self.pos = pos
        self.c_type = None
        self.placeholder = None

    def __str__(self):
        if not self.placeholder:
            return f"Cell({self.pos})"
        else:
            return str(self.placeholder)

    @staticmethod
    def _manhattan_distance(pos1: Pos, pos2: Pos) -> int:
        """Calculates the manhattan distance between two positions

        Manhattan distance ((x1,y1), (x2,y2)) = |x1 - x2| + |y1 - y2|

        Args:
            pos1 (Pos): First position
            pos2 (Pos): Second position

        Returns:
            int: Manhattan distance between the points
        """
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)

    def _is_free(self) -> bool:
        """Checks if cell is free

        Returns:
            bool: True if cell is free
        """
        return not self.c_type

    def distance_to(self, new_cell: "Cell") -> int:
        """Distance between current cell and corresponding cell

        Args:
            new_cell (Cell): Corresponding Cell

        Returns:
            int: Distance between the cells
        """
        curr_cell_pos = self.pos
        new_cell_pos = new_cell.pos

        return self._manhattan_distance(curr_cell_pos, new_cell_pos)

    def assign(self, object: Any, exists_ok: bool = False):
        """Assign an object to a cell

        Args:
            object (Any): Object to be assigned
            exists_ok (bool, optional): Assign whether the cell is not empty. Defaults to False.

        Raises:
            ValueError: If the cell is not empty and exists_ok is false
        """
        if not self._is_free():
            if exists_ok:
                self.clear()
            else:
                raise ValueError(f"Position : {self.pos} is already occupied")

        self.c_type = object.__class__.__name__
        self.placeholder = object
        setattr(self.placeholder, "pos", self.pos)

    def clear(self) -> Any:
        """Clears the current cell

        Returns:
            Any: The placeholder of the cell, None if nothing found
        """
        place_holder = self.placeholder
        self.c_type = None
        self.placeholder = None

        return place_holder
