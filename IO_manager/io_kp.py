import os
import numpy as np
from numpy.core._multiarray_umath import ndarray


distributions = [ "uncorrelated",
                  "weakly_correlated",
                  "strongly_correlated",
                  "inverse_strongly_correlated",
                  "subset_sum",
                  "multiple_strongly_correlated",
                  "profit_ceiling",
                  "circle"]


class KP_Instance_Creator:
    nItems: int
    distribution: str
    capacity: int
    volume_items: ndarray
    profit_items: ndarray
    existing_distributions = distributions

    def __init__(self, mode, seed=1, dimension=50):
        self.seed_ = seed
        np.random.seed(self.seed_)
        if mode == "random":
            self.my_random(dimension=dimension)
        else:
            self.read_data(mode)
        self.distribution = mode

    def read_data(self, name_type):
        assert name_type in self.existing_distributions, f"the distribution {name_type} does not exits"
        folder = "problems/KP/"
        if "AI" not in os.getcwd():
            folder = "AI2020/problems/TSP/"
        files_distr = [file_ for file_ in os.listdir(folder) if name_type in file_]
        file_object = np.random.choice(files_distr, 1)[0]
        file_object = open(f"{folder}{file_object}")
        data = file_object.read()
        file_object.close()
        lines = data.splitlines()

        self.nItems = int(lines[0])
        self.capacity = int(lines[1])

        self.volume_items = np.zeros(self.nItems)
        self.profit_items = np.zeros(self.nItems)
        for i in range(self.nItems):
            line_i = lines[3 + i].split(' ')
            self.profit_items[i] = int(line_i[0])
            self.volume_items[i] = int(line_i[1])

    def my_random(self, dimension=50):
        self.volume_items = np.random.uniform(0, 200, dimension).astype(np.int)
        self.profit_items = np.random.uniform(0, 200, dimension).astype(np.int)
        num_items_prob = np.random.choice(np.arange(1, dimension//2), 1)
        self.max_cost = np.cast(np.mean(self.volume_items) * num_items_prob, np.int)

    def plot_data_scatter(self):
        pass

    def plot_data_distribution(self):
        pass
