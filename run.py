from IO_manager.io_tsp import TSP_Instance_Creator
from IO_manager.io_kp import KP_Instance_Creator, distributions
from IO_manager.generate_new_instances import generate_instance
from solvers.TSP_solver import Solver_TSP
from solvers.constructive_algorithms import Random_Initializer
from solvers.local_search import TwoOpt
import numpy as np
import pandas as pd


def run_trial_TSP():
    ic = TSP_Instance_Creator("standard", name_problem="eil76.tsp")
    ic.print_info()
    ic.plot_data()
    # ic.plot_solution()
    # print(ic.dist_matrix)

    seeds = [0, 123, 333]
    time_to_solve = 1  # in seconds

    names_instances = ["d198.tsp"]  # , "pr439.tsp", "u1060.tsp"
    samples = {name: {} for name in names_instances}
    two_opt_results = []
    collectors = [two_opt_results]
    initializers = ["random"]
    init_functions = [Random_Initializer.random_method]
    improvements = ["2 opt"]
    improve_functions = [TwoOpt.local_search]
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
