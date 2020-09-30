from IO_manager.io_tsp import TSP_Instance_Creator
from IO_manager.io_kp import KP_Instance_Creator, distributions

def run_trial_TSP():
    ic = TSP_Instance_Creator("standard", name_problem="eil76.tsp")
    ic.print_info()
    ic.plot_data()
    ic.plot_solution()
    print(ic.dist_matrix)

def run_trial_KP():
    ic = KP_Instance_Creator("random")
    print(distributions)

if __name__ == '__main__':
    run_trial_KP()