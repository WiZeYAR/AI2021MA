from numpy.core._multiarray_umath import ndarray
import os
from time import time as t
import matplotlib.pyplot as plt
if 'AI' in os.getcwd():
    from solvers.constructive_algorithms import *
    from solvers.local_search import *
else:
    from AI2020.solvers.constructive_algorithms import *
    from AI2020.solvers.local_search import *


class Solver_TSP:

    solution: ndarray
    found_length: float
    available_initializers = {"random": random_initialier.random_method,
                              "nearest_neighbors": nearest_neighbor.nn,
                              # "best_nn": nearest_neighbor.best_nn,
                              "multi_fragment": multi_fragment.mf
                              }

    available_improvements = {"2-opt": TwoOpt.local_search,
                              "2.5-opt": TwoDotFiveOpt.local_search}



    def __init__(self, initializer, seed_=0, stop_run_after=180):
        self.initializer = initializer[0]
        self.methods_name = [initializer[0]]
        self.methods = [initializer[1]]
        self.name_method = "initialized with " + self.initializer
        self.solved = False
        self.seed = seed_
        self.max_time = stop_run_after
        # assert self.initializer in self.available_initializers, f"the {initializer} initializer is not available currently."

    def bind(self, local_or_meta):
        # assert local_or_meta in self.available_improvements, f"the {local_or_meta} method is not available currently."
        self.methods.append(local_or_meta[1])
        self.methods_name.append(local_or_meta[0])
        self.name_method += ", improved with " + local_or_meta[0]

    def pop(self):
        self.methods.pop()
        self.name_method = self.name_method[::-1][self.name_method[::-1].find("improved"[::-1]) + len("improved") + 2:][
                           ::-1]

    def __call__(self, instance_, verbose=False, return_value=False):
        self.instance = instance_
        self.solved = False
        self.ls_calls = 0
        if verbose:
            print(f"###  solving with {self.methods} ####")
        start = t()
        self.solution = self.methods[0](instance_.dist_matrix)
        # assert self.check_if_solution_is_valid(self.solution), "Error the solution is not valid"
        for i in range(1, len(self.methods)):
            self.solution, ls = self.methods[i](self.solution, self.instance.dist_matrix)
            self.ls_calls += ls
            # assert self.check_if_solution_is_valid(self.solution), "Error the solution is not valid"
            if t() - start > self.max_time:
                break

        end = t()
        self.time_to_solve = np.around(end - start,3)
        self.solved = True
        self.evaluate_solution()
        self._gap()
        if verbose:
            print(f"###  solution found with {self.gap} % gap  in {self.time_to_solve} seconds ####")
            print(f"the total length for the solution found is {self.found_length}",
                  f"while the optimal length is {self.instance.best_sol}",
                  f"the gap is {self.gap}%",
                  f"the number of LS calls are {self.ls_calls}",
                  f"the solution is found in {self.time_to_solve} seconds", sep="\n")

        if return_value:
            return self.solution

    def plot_solution(self):
        assert self.solved, "You can't plot the solution, you need to solve it first!"
        plt.figure(figsize=(8, 8))
        self._gap()
        plt.title(f"{self.instance.name} solved with {self.name_method} solver, gap {self.gap}")
        ordered_points = self.instance.points[self.solution]
        plt.plot(ordered_points[:, 1], ordered_points[:, 2], 'b-')
        plt.show()

    def check_if_solution_is_valid(self, solution):
        rights_values = np.sum([self.check_validation(i, solution) for i in np.arange(self.instance.nPoints)])
        # print(rights_values, self.instance.nPoints)
        if rights_values == self.instance.nPoints:
            return True
        else:
            return False

    def check_validation(self, node, solution):
        if np.sum(solution == node) == 1:
            return 1
        else:
            return 0

    def evaluate_solution(self, return_value=False):
        total_length = 0
        starting_node = self.solution[0]
        from_node = starting_node
        for node in self.solution[1:]:
            total_length += self.instance.dist_matrix[from_node, node]
            from_node = node

        self.found_length = total_length
        if return_value:
            return total_length

    def _gap(self):
        self.evaluate_solution(return_value=False)
        self.gap = np.round(((self.found_length - self.instance.best_sol) / self.instance.best_sol) * 100, 2)
