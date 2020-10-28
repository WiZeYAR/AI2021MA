import numpy as np
import matplotlib.pyplot as plt


def DJ1(x):
    return sum(x)

def DJ2(x):
    return sum(x)

def DJ3(x):
    return sum(x)

def DJ4(x):
    return sum(x)

def DJ5(x):
    return sum(x)



class De_Jong:
    dimension: int
    function_name: str
    range: (float, float)
    resolution_factor: float
    num_bits: int
    funcs = {1:('De jong 1', DJ1, 5.12, 0.01, 3 ),
             2:('De jong 2', DJ2, 2.048, 0.001, 2 ),
             3:('De jong 3', DJ3, 5.12, 0.01, 4),
             4:('De jong 4', DJ4, 1.28, 0.01, 30),
             5:('De jong 5', DJ5, 65.536, 0.001, 2)}

    def __init__(self, func_number, dimension_in=False):
        assert isinstance(func_number, int), f'func_number needs to be an integer!!!'
        assert func_number < 5, 'func_number needs to be a number between 1 and 5'
        self.function_name, self.fun, min_, self.resolution_factor, self.dimension = self.funcs[func_number]
        self.range = (-min_ , min_ - self.resolution_factor)
        self.num_bits = int(np.log2(int(2* (min_/ self.resolution_factor))))
        if dimension_in:
            self.dimension = dimension_in


    def evaluate(self, x_e):
        x = De_Jong.gray_decode(x_e)
        return self.fun(x)

    @staticmethod
    def gray_encode(n):
        return n ^ n >> 1

    @staticmethod
    def gray_decode(n):
        m = n >> 1
        while m:
            n ^= m
            m >>= 1
        return n
