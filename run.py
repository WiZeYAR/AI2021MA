from IO_manager.io_tsp import TSP_Instance_Creator

def run_trial():
    ic = TSP_Instance_Creator("standard")
    ic.print_info()
    ic.plot_data()
    print(ic.dist_matrix)


if __name__ == '__main__':
    run_trial()