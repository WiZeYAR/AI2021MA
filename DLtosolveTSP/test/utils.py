import numpy as np
import matplotlib.pyplot as plt


def plot_solution(pos, tour, name, color):
    plot_points(pos, tour)
    trip = pos[tour]
    tour = np.array(list(range(len(trip))) + [0])  # Plot tour
    X = trip[tour, 0]
    Y = trip[tour, 1]
    plt.plot(X, Y, color)
    plt.title(name)
    plt.show()





def plot_points(pos, tour_found):
    plt.scatter(pos[:, 0], pos[:, 1], color='gray')
    trip = pos[tour_found]
    z, y = trip[:, 0], trip[:, 1]
    plt.scatter(z, y, color='b')
    for i, txt in enumerate(tour_found[:-1]):  # tour_found[:-1]
        plt.annotate(txt, (z[i], y[i]))


def evaluate_solution(solution, dist):
    total_length = 0
    starting_node = solution[0]
    from_node = starting_node
    dist_matrix = np.copy(dist)
    dist_matrix = np.round(dist_matrix * 10000, 0)
    for node in solution[1:]:
        total_length += dist_matrix[from_node, node]
        from_node = node

    total_length += dist_matrix[from_node, starting_node]
    return total_length / 10000
