import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def DJ1(x):
    return np.sum(x**2)

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

    def gray_encode(self, n):
        val = n ^ n >> 1
        r_val = f"{val:>b}"
        pad = "0"*(self.num_bits - len(r_val))
        return pad + r_val

    @staticmethod
    def gray_decode(n_s):
        n = int(n_s, 2)
        m = n >> 1
        while m:
            n ^= m
            m >>= 1
        return n

    def plot(self):
        samples=100
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        x = np.linspace(self.range[0], self.range[1], samples)
        y = np.linspace(self.range[0], self.range[1], samples)

        X, Y = np.meshgrid(x, y)
        z_ = np.stack([X.flatten(), Y.flatten()], axis=-1)
        Z_ = np.apply_along_axis(self.fun, -1, z_)
        Z = np.resize(Z_, (samples, samples))

        # ax.plot_wireframe(X, Y, Z, color='green')
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        cset = ax.contour(X, Y, Z, zdir='z', cmap=cm.coolwarm)
        cset = ax.contour(X, Y, Z, zdir='x', cmap=cm.coolwarm)
        cset = ax.contour(X, Y, Z, zdir='y', cmap=cm.coolwarm)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y) ')

        plt.show()