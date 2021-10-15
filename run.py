from IO_manager.io_tsp import TSP_Instance_Creator
from IO_manager.io_kp import KP_Instance_Creator, distributions
from IO_manager.generate_new_instances import generate_instance
from solvers.TSP_solver import Solver_TSP
from solvers.constructive_algorithms import Random_Initializer
from solvers.local_search import TwoOpt
import numpy as np
import pandas as pd


class DoubleBridge:

    @staticmethod
    def difference_cost(solution, a, b, c, d, matrix):
        n = matrix.shape[0]
        to_remove = matrix[solution[a - 1], solution[a]] + matrix[solution[b - 1], solution[b]] + matrix[
            solution[c - 1], solution[c]] + matrix[solution[d - 1], solution[d]]
        to_add = matrix[solution[a], solution[c - 1]] + matrix[solution[b], solution[d - 1]] + matrix[
            solution[c], solution[a - 1]] + matrix[solution[d], solution[b - 1]]
        return to_add - to_remove

    @staticmethod
    def perturbate_solution(solution, actual_cost, matrix):
        a, b, c, d = np.sort(np.random.choice(matrix.shape[0], size=4))
        # print(a,b,c,d)
        A = solution[a:b]
        B = solution[b:c]
        C = solution[c:d]
        D = np.concatenate((solution[d:], solution[:a]))
        new_solution = np.concatenate((D, C, B, A))
        new_length = actual_cost + DoubleBridge.difference_cost(solution, a, b, c, d, matrix)
        return new_solution, new_length


cost_sol_better = []


def my_local_search(solution, cost_sol, dist_matrix):
    new_solution, new_cost = solution, cost_sol
    for data in TwoOpt.local_search(solution, cost_sol, dist_matrix):
        new_solution, new_cost, numbert_calls_LS, ending_cond = data
    return new_solution, new_cost, 1, True


class ILS_better:
    @staticmethod
    def solve(solution, actual_cost, matrix):
        global cost_sol_better
        new_sol, new_cost, ls_c, _ = my_local_search(solution, actual_cost, matrix)
        best_sol, best_cost = new_sol, new_cost
        cost_sol_better.append(new_cost)
        calls = 0
        while True:
            calls += 1
            new_sol, new_cost = DoubleBridge.perturbate_solution(best_sol, best_cost, matrix)
            new_sol, new_cost, _, _ = my_local_search(new_sol, new_cost, matrix)
            cost_sol_better.append(new_cost)
            if new_cost < best_cost:
                best_sol, best_cost = new_sol, new_cost

            yield best_sol, best_cost, calls, False


def run_trial_TSP():
    ic = TSP_Instance_Creator("standard", name_problem="eil76.tsp")
    ic.print_info()
    # ic.plot_data()
    # ic.plot_solution()
    # print(ic.dist_matrix)

    seeds = [0, 123, 333]
    time_to_solve = 10  # in seconds

    names_instances = ["d198.tsp"]  # , "pr439.tsp", "u1060.tsp"
    samples = {name: {} for name in names_instances}
    two_opt_results, ils_results = [], []
    collectors = [two_opt_results, ils_results]
    initializers = ["random"]
    init_functions = [Random_Initializer.random_method]
    improvements = ["2 opt", "ILS better"]
    improve_functions = [TwoOpt.local_search, ILS_better.solve]
    results = []
    index = []
    for s_ in seeds:
        for i, init in enumerate(initializers):
            for j, improve in enumerate(improvements):
                solver = Solver_TSP((init, init_functions[i]), seed_=s_, stop_run_after=time_to_solve)
                solver.bind((improve, improve_functions[j]))
                for name in names_instances:
                    instance = TSP_Instance_Creator("standard", name)
                    solver(instance)
                    index.append((name, instance.best_sol, solver.name_method, s_))
                    results.append([solver.found_length, solver.gap, solver.time_to_solve, solver.ls_calls])
                    samples[name][improve] = np.round(np.abs(np.array(collectors[j]) - ic.best_sol) / ic.best_sol * 100,
                                                      2)
                    # if j == 0:
                    #   cost_sol_better = []
                    # elif j == 1:
                    #   cost_sol_RW = []
                    # else:
                    #   cost_sol_LSMC = []

    index = pd.MultiIndex.from_tuples(index, names=['problem', 'optimal lenght', 'method', 'seed'])

    df = pd.DataFrame(results, index=index, columns=["tour length", "gap", "time to solve", "calls Local Search"])
    print(df.head())


def run_trial_KP():
    # print("random")
    ic = KP_Instance_Creator("random")
    ic.plot_data_scatter()
    ic.plot_data_distribution()
    for i in range(len(distributions)):
        # print(distributions[i])
        ic = KP_Instance_Creator(distributions[i])
        ic.plot_data_scatter()
        ic.plot_data_distribution()
    # i = 7
    # print(distributions[i])
    # ic = KP_Instance_Creator(distributions[i])
    # ic.plot_data_scatter()
    # ic.plot_data_distribution()


if __name__ == '__main__':
    # run_trial_KP()
    # generate_instance()
    run_trial_TSP()
