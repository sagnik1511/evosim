from evosim.elements.elems import Resource


class Wood(Resource):

    def __init__(self, pos, hp: int = 1, max_hp: int = 10, *args):
        super().__init__(pos=pos, hp=hp, max_hp=max_hp, *args)
        self.grow_delta: int = 0.1

    def __str__(self):
        return f"Wood({self.pos}, Resources={self.curr_res_amount})"
