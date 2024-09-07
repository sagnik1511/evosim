from evosim.elements.elems import Resource


class Wood(Resource):

    def __init__(self, pos, *args, hp: int = 1, max_hp: int = 10):
        super().__init__(pos=pos, hp=hp, max_hp=max_hp, *args)
        self.grow_delta: int = 0.0001

    def __str__(self):
        return f"Wood({self.pos}, Resources={self.hp})"
