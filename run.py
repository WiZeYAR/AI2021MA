from IO_manager.io_tsp import TSP_Instance_Creator
from IO_manager.io_kp import KP_Instance_Creator, distributions
from IO_manager.generate_new_instances import generate_instance

def run_trial_TSP():
    ic = TSP_Instance_Creator("standard", name_problem="eil76.tsp")
    ic.print_info()
    ic.plot_data()
    ic.plot_solution()
    print(ic.dist_matrix)

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
    generate_instance()