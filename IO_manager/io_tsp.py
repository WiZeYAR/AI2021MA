import numpy as np
from typing import List
from matplotlib import pyplot as plt
from numpy.core._multiarray_umath import ndarray


class Instance:
    nPoints: int
    best_sol: int
    name: str
    lines: List[str]
    dist_matrix: ndarray
    points: ndarray

    def __init__(self, name_tsp):
        self.read_instance(name_tsp)

    def read_instance(self, name_tsp):
        # read raw data
        file_object = open(name_tsp)
        data = file_object.read()
        file_object.close()
        self.lines = data.splitlines()

        # store data set information
        self.name = self.lines[0].split(' ')[2]
        self.nPoints = np.int(self.lines[3].split(' ')[2])
        self.best_sol = np.float(self.lines[5].split(' ')[2])

        # read all data points and store them
        self.points = np.zeros((self.nPoints, 3))
        for i in range(self.nPoints):
            line_i = self.lines[7 + i].split(' ')
            self.points[i, 0] = int(line_i[0])
            self.points[i, 1] = line_i[1]
            self.points[i, 2] = line_i[2]

        self.create_dist_matrix()
        self.exist_opt = False
        if [name for name in ["eil76", "kroA100"] if name in name_tsp]:
            self.exist_opt = True
            file_object = open(name_tsp.replace(".tsp", ".opt.tour"))
            data = file_object.read()
            file_object.close()
            lines = data.splitlines()

            # read all data points and store them
            self.optimal_tour = np.zeros(self.nPoints, dtype=np.int)
            for i in range(self.nPoints):
                line_i = lines[5 + i].split(' ')
                self.optimal_tour[i] = int(line_i[0]) - 1

    def print_info(self):
        print('name: ' + self.name)
        print('nPoints: ' + str(self.nPoints))
        print('best_sol: ' + str(self.best_sol))

    def plot_data(self):
        plt.figure(figsize=(8, 8))
        plt.title(self.name)
        plt.scatter(self.points[:, 1], self.points[:, 2])
        for i, txt in enumerate(np.arange(self.nPoints)):  # tour_found[:-1]
            plt.annotate(txt, (self.points[i, 1], self.points[i, 2]))
        plt.show()

    @staticmethod
    def distance_euc(zi, zj):
        xi, xj = zi[0], zj[0]
        yi, yj = zi[1], zj[1]
        return round(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 0)

    def create_dist_matrix(self):
        self.dist_matrix = np.zeros((self.nPoints, self.nPoints))

        for i in range(self.nPoints):
            for j in range(i, self.nPoints):
                self.dist_matrix[i, j] = self.distance_euc(self.points[i][1:3], self.points[j][1:3])
        self.dist_matrix += self.dist_matrix.T

