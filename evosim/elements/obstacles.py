from evosim.elements.elems import Obstacle


class Rock(Obstacle):

    def __init__(self, pos):
        super().__init__(pos)

    def detoriate(self):
        """Rock doesn't detoriate"""
        pass

    def __str__(self):
        return f"Rock({self.pos})"
