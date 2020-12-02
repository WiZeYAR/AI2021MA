import numpy as np
import os, tempfile
import pandas as pd
from tqdm import tqdm
from time import time as t
import matplotlib.pyplot as plt

from concorde.tsp import TSPSolver
from DLtosolveTSP.test.utils import evaluate_solution
from DLtosolveTSP.test.Good_Edge_Values import GoodEdgeValues
from DLtosolveTSP.test.reconstruction_algorithm import Multi_Fragment, TwoOpt, TwoOpt_for_GEV
from DLtosolveTSP.test.tsplib_reader import Generate_Data_for_TSPLib_files


def results_on_tsplib(settings, epoch='', iteration_train=''):
    generator_data = Generate_Data_for_TSPLib_files(settings)

    results, index_ = [], []
    iteration = 0
    df_losses = {}
    for data in tqdm(generator_data.create_data(), total=len(generator_data.reader.files)):
        # Concorde
        optimal_tour = optimal_solver(data[1], data[4])
        # print(optimal_tour)
        checker = test_checker(data, results, index_, "Concorde", optimal_tour, 0)

        # Greedy MF
        solver_MF = Multi_Fragment(data[2], data[1], if_distance=True)
        checker(data, results, index_, "greedy", solver_MF.solution, solver_MF.time)

        # 2opt on greedy sol
        solution_2opt, time_2opt = TwoOpt.loop2opt(solver_MF.solution, data[2])
        checker(data, results, index_, "2opt", solution_2opt, solver_MF.time + time_2opt)
        start_creating_image = t()
        gev_handler = GoodEdgeValues(settings)
        gev_handler.create_GEV(data)
        time_for_gev = t() - start_creating_image

        df_losses[data[4]] = gev_handler.loss

        # greedy on GEV
        solver_MF_and_CNN = Multi_Fragment(gev_handler.gev, gev_handler.pos,
                                           to_plot=False, distance_matrix=data[2])
        checker(data, results, index_, f"EV-greedy", #  {image_filter}  {filter_edge}
                solver_MF_and_CNN.solution, solver_MF_and_CNN.time + time_for_gev)

        # greedy EV followed by 2opt
        solution_ev_2opt, time_ev_2opt = TwoOpt.loop2opt(solver_MF_and_CNN.solution, data[2])
        checker(data, results, index_, f"EV-greedy+2opt",
                solution_ev_2opt, solver_MF_and_CNN.time + time_ev_2opt + time_for_gev)

        # # EV-2opt
        solution_ev_2opt_gev, time_ev_2opt_gev = TwoOpt_for_GEV.loop2opt(solver_MF_and_CNN.solution, gev_handler.gev)
        checker(data, results, index_, f"EV-2opt",
                solution_ev_2opt_gev, solver_MF_and_CNN.time + time_ev_2opt_gev + time_for_gev)

        # # EV-2opt + 2opt
        solution_ev_2opt_gev_2opt, time_ev_2opt_gev_2opt = TwoOpt.loop2opt(solution_ev_2opt_gev, data[2])
        checker(data, results, index_, f"EV-2opt+2opt",
                solution_ev_2opt_gev_2opt, solver_MF_and_CNN.time + time_ev_2opt_gev_2opt
                + time_ev_2opt_gev + time_for_gev)


    index_ = pd.MultiIndex.from_tuples(index_, names=["name instance", 'method'])
    df = pd.DataFrame(results, index=index_, columns=["tour length", "gap from the optimal", "time to solve"])

    df_pivot = df.reset_index()
    df_pivot = df_pivot.pivot(index="name instance", columns="method", values="gap from the optimal")
    df_pivot = df_pivot.loc[[name[:-4] for name in generator_data.reader.files],
                            ["EV-greedy",	"EV-greedy+2opt", "EV-2opt", "EV-2opt+2opt"]] #, "greedy", "2opt"] ]
    df_pivot['loss'] = pd.Series(df_losses)
    print(df_pivot)
    df_pivot.to_csv(f"./data/tsplib_problems/gaps_different_methods_ottimo{['_noMST', '_with_MST'][settings.with_MST]}"
                    f"{['', '_modified'][settings.modify_heuristics]}.csv")

    df_mean_pivot = df_pivot.mean(axis=0)
    if isinstance(epoch, int):
        folder = f"{['', 'filtered/'][settings.filtered]}{['noMST/', 'withMST/'][settings.with_MST]}"
        df_mean_pivot.to_csv(f"./train/train_infos/{folder}{epoch}_{iteration_train}"
                             f"{['_noMST', '_with_MST'][settings.with_MST]}"
                             f"{['', '_modified'][settings.modify_heuristics]}_tsplib.csv", header=False)
        if df_mean_pivot.iloc[1] < 2:
            df_pivot.to_csv(f"./train/train_infos/{folder}risultati/{epoch}_{iteration_train}_tsp.csv")
        return df_mean_pivot.iloc[1]



class test_checker:

    def __init__(self, data, res, ind_, name_solver, tour, time_to_solve):
        self.pos, dist, name_instance = data[1], data[2], data[4]
        self.opt_tour = np.append(tour, tour[0])
        self.optimal_len = evaluate_solution(self.opt_tour, dist)
        res.append([self.optimal_len, 0.000, time_to_solve])
        ind_.append([name_instance, name_solver])

    def __call__(self, data, res, ind_, name_solver, tour, time_to_solve):
        dist, name_instance = data[2], data[4]
        len_ = evaluate_solution(tour, dist)
        gap_ = (len_ - self.optimal_len)/ self.optimal_len * 100
        if gap_ < 0:
            print(f"\n{name_solver}\n"
                  f"check len tour    : {len(tour)} \n"
                  f"check optimal tour: {len(self.opt_tour)}\n"
                  f"check optimal gap : {self.optimal_len} \n"
                  f"check actual len  : {len_}")
        res.append([len_, gap_, time_to_solve])
        ind_.append([name_instance, name_solver])



def optimal_solver(pos, name_instance):
    dir = os.getcwd()
    with tempfile.TemporaryDirectory() as path:
        os.chdir(path)
        start = t()
        solver = TSPSolver.from_data(
            pos[:, 0],
            pos[:, 1],
            norm="EUC_2D"
        )
        solution = solver.solve()
    os.chdir(dir)
    np.save(f"./problems/TSP/optimal/{name_instance}.npy", solution.tour)
    return solution.tour
