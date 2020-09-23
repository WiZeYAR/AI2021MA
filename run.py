from IO_manager import *
import pandas as pd


def add(solver, instance, improve, index, results, name, verbose, show_plots):
    solver.bind(improve)
    solver(instance, return_value=False, verbose=verbose)

    if verbose:
        print(f"the total length for the solution found is {solver.found_length}",
              f"while the optimal length is {instance.best_sol}",
              f"the gap is {solver.gap}%",
              f"the solution is found in {solver.time_to_solve} seconds", sep="\n")

    index.append((name, solver.name_method))
    results.append([solver.found_length, instance.best_sol, solver.gap, solver.time_to_solve])

    if show_plots:
        solver.plot_solution()


def run(show_plots=False, verbose=False):
    # names = [name_ for name_ in os.listdir("./problems") if "tsp" in name_]
    names = ["eil76.tsp"]
    initializers = []  #Solver_TSP.available_initializers.keys()
    improvements = ['simulated_annealing']  #Solver_TSP.available_improvements.keys()
    results = []
    index = []
    for name in names:
        filename = f"problems/{name}"
        instance = Instance(filename)
        if verbose:
            print("\n\n#############################")
            instance.print_info()
        if show_plots:
            instance.plot_data()

        for init in initializers:
            for improve in improvements:
                solver = Solver_TSP(init)
                add(solver, instance, improve, index, results, name, verbose, show_plots)
                # for improve2 in [j for j in improvements if j not in [improve]]:
                #     add(solver, instance, improve2, index, results, name, verbose, show_plots)
                #
                #     for improve3 in [j for j in improvements if j not in [improve, improve2]]:
                #         add(solver, instance, improve3, index, results, name, verbose, show_plots)
                #         solver.pop()
                #
                #     solver.pop()


        if instance.exist_opt and show_plots:
            solver.solution = np.concatenate([instance.optimal_tour, [instance.optimal_tour[0]]])
            solver.method = "optimal"
            solver.plot_solution()

    index = pd.MultiIndex.from_tuples(index, names=['problem', 'method'])

    return pd.DataFrame(results, index=index, columns=["tour length", "optimal solution", "gap", "time to solve"])


if __name__ == '__main__':
    df = run(show_plots=False, verbose=True)
    df.to_csv("./results.csv")
