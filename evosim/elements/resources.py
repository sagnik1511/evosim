from evosim.elements.elems import Resource


class Wood(Resource):

    def __init__(self, pos):
        super().__init__(pos=pos, hp=100)
        self.grow_delta: int = 100

    def __str__(self):
        return f"Wood({self.pos}, Resources={self.curr_res_amount})"
