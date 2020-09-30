import os
import numpy as np
from numpy.core._multiarray_umath import ndarray

class KP_Instance_Creator:
    nItems: int
    distribution: str
    capacity: int
    volume_items: ndarray
    profit_items: ndarray
    existing_distributions = ["uncorrelated",
                              "weakly_correlated",
                              "strongly_correlated",
                              "inverse_strongly_correlated",
                              "subset_sum",
                              "multiple_strongly_correlated",
                              "profit_ceiling",
                              "circle"]

    def __init__(self, mode, distribution=False, seed=1, dimension=50):
        self.seed_ = seed
        np.random.seed(self.seed_)
        if mode == "my_random":
            self.my_random(dimension=dimension)
        else:
            self.read_data(mode)

    def read_data(self, name_type):
        assert name_type in self.existing_distributions, f"the distribution {name_type} does not exits"
        folder = "problems/KP/"
        if "AI" not in os.getcwd():
            folder = "AI2020/problems/TSP/"
        files_distr = [file_ for file_ in os.listdir(folder) if name_type in file_]
        file_object = np.random.choice(files_distr, 1)
        file_object = open(f"{folder}{file_object}")
        data = file_object.read()
        file_object.close()
        self.lines = data.splitlines()

    def my_random(self, dimension=50):
        self.volume_items = np.random.uniform(0, 200, dimension)
        self.profit_items = np.random.uniform(0, 200, dimension)
        num_items_prob = np.random.choice(np.arange(1, dimension//2), 1)
        self.max_cost = np.mean(self.weights) * num_items_prob

    def plot_data_scatter(self):
        pass

    def plot_data_distribution(self):
        pass
