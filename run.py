from IO_manager.io_tsp import TSP_Instance_Creator

def run_trial():
    ic = TSP_Instance_Creator("standard", name_problem="eil76.tsp")
    ic.print_info()
    ic.plot_data()
    ic.plot_solution()
    print(ic.dist_matrix)


if __name__ == '__main__':
    run_trial()