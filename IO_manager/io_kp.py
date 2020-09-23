import numpy as np

class KP_Instance_Creator:
    def __init__(self, dimension):
        self.weights = np.random.uniform(0, 200, dimension)
        self.values = np.random.uniform(0, 200, dimension)
        num_items_prob = np.random.choice(np.arange(1, dimension//2), 1)
        self.max_cost = np.mean(self.weights) * num_items_prob

    def plot_data(self):
        pass
