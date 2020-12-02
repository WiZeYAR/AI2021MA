import numpy as np
from time import time as t
import matplotlib.pyplot as plt

class Multi_Fragment:
    def __init__(self, good_edge_values, pos, to_plot=False, if_distance=False, distance_matrix = None):
        self.number_of_cities = good_edge_values.shape[0]
        self.pos = np.copy(pos)
        self.to_plot = to_plot
        self.mat = np.copy(good_edge_values)
        self.distance_mat = if_distance
        self.dist_matrix = np.copy(distance_matrix)
        start = t()
        self.solution = self.create_solution()
        self.time = t() - start

    def create_solution(self):
        solution = {str(i): [] for i in range(self.number_of_cities)}
        start_list = [i for i in range(self.number_of_cities)]
        inside = 0
        if self.distance_mat:
            lower_indices = np.tril_indices(self.mat.shape[0])
            self.mat[lower_indices] = np.infty
            iterator = np.argsort(self.mat.flatten())
        else:
            lower_indices = np.tril_indices(self.mat.shape[0])
            diag_indices = np.diag_indices(self.mat.shape[0])
            self.mat[diag_indices] = 0
            iterator = np.argsort(self.mat.flatten())[::-1]
            self.dist_matrix[lower_indices] = np.infty
        for el in iterator:
            node1, node2 = el // self.number_of_cities, el % self.number_of_cities
            possible_edge = [node1, node2]
            if Multi_Fragment.check_if_available(node1, node2,
                                                 solution):
                if Multi_Fragment.check_if_not_close(possible_edge, solution):
                    if self.mat[node1, node2] == 0 or node2==node1:
                        # print(self.mat[node1, node2], node2, node1)
                        solution = self.complete_solution_with_distances(start_list, solution, inside)
                        return solution
                    solution[str(node1)].append(node2)
                    solution[str(node2)].append(node1)
                    if len(solution[str(node1)]) == 2:
                        start_list.remove(node1)
                    if len(solution[str(node2)]) == 2:
                        start_list.remove(node2)
                    inside += 1

                    if inside == self.number_of_cities - 1:
                        solution = Multi_Fragment.construct_solution(start_list, solution, self.number_of_cities)
                        return solution

    def complete_solution_with_distances(self, start_list, solution, ins):
        inside = ins
        for el in np.argsort(self.dist_matrix.flatten()):
            node1, node2 = el // self.number_of_cities, el % self.number_of_cities
            possible_edge = [node1, node2]
            if Multi_Fragment.check_if_available(node1, node2,
                                                 solution):
                if Multi_Fragment.check_if_not_close(possible_edge, solution):
                    solution[str(node1)].append(node2)
                    solution[str(node2)].append(node1)
                    if len(solution[str(node1)]) == 2:
                        start_list.remove(node1)
                    if len(solution[str(node2)]) == 2:
                        start_list.remove(node2)
                    inside += 1
                    if self.to_plot:
                        print(f"nodi aggiunti {possible_edge} con valore distanza {self.dist_matrix[node1, node2]}")
                        self.plot_partial_sol(solution)
                        input()

                    if inside == self.number_of_cities - 1:
                        solution = Multi_Fragment.construct_solution(start_list, solution, self.number_of_cities)
                        return solution


    @staticmethod
    def construct_solution(start_sol, sol, n):
        assert len(start_sol) == 2, "too many cities with just one link"
        end = False
        n1, n2 = start_sol
        from_city = n2
        sol_list = [n1, n2]
        iterazione = 0
        while not end:
            for node_connected in sol[str(from_city)]:
                iterazione += 1
                if node_connected not in sol_list:
                    from_city = node_connected
                    sol_list.append(node_connected)

                if iterazione > 300:
                    if len(sol_list) == n:
                        end = True
        sol_list.append(n1)
        return sol_list

    @staticmethod
    def check_if_available(n1, n2, sol):
        if len(sol[str(n1)]) < 2 and len(sol[str(n2)]) < 2:
            return True
        else:
            return False

    @staticmethod
    def check_if_not_close(edge_to_append, sol):
        n1, n2 = edge_to_append
        from_city = n2
        if len(sol[str(from_city)]) == 0:
            return True
        partial_tour = [from_city]
        end = False
        iterazione = 0
        return_value = True
        while not end:
            if len(sol[str(from_city)]) == 1:
                if from_city == n1:
                    return_value = False
                    end = True
                elif iterazione > 1:
                    return_value = True
                    end = True
                else:
                    from_city = sol[str(from_city)][0]
                    partial_tour.append(from_city)
                    iterazione += 1
            else:
                for node_connected in sol[str(from_city)]:
                    if node_connected not in partial_tour:
                        from_city = node_connected
                        partial_tour.append(node_connected)
                        iterazione += 1
        return return_value


    def plot_partial_sol(self, solution_dict):
        plt.axis('equal')
        plt.scatter(self.pos[:, 0], self.pos[:, 1], color='gray')
        for key in solution_dict.keys():
            if len(solution_dict[key])>0:
                for j in solution_dict[key]:
                    nodes = [j , int(key)]
                    plt.plot(self.pos[nodes,0],
                             self.pos[nodes,1], "b-")
        plt.show()




class TwoOpt:
    @staticmethod
    def step2opt(solution, matrix_dist, distance):
        seq_length = len(solution) - 1
        tsp_sequence = np.array(solution)
        uncrosses = 0
        for i in range(1, seq_length):
            for j in range(i + 1, seq_length):
                new_tsp_sequence = TwoOpt.swap2opt(tsp_sequence, i, j)
                new_distance = distance + TwoOpt.gain(i, j, tsp_sequence, matrix_dist)
                if new_distance < distance:
                    uncrosses += 1
                    tsp_sequence = np.copy(new_tsp_sequence)
                    distance = new_distance
        return tsp_sequence, distance, uncrosses

    @staticmethod
    def swap2opt(tsp_sequence, i, j):
        new_tsp_sequence = np.copy(tsp_sequence)
        new_tsp_sequence[i:j + 1] = np.flip(tsp_sequence[i:j + 1], axis=0)  # flip or swap ?
        return new_tsp_sequence

    @staticmethod
    def gain(i, j, tsp_sequence, matrix_dist):
        old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[i - 1]] + matrix_dist[
            tsp_sequence[j], tsp_sequence[j + 1]])
        changed_links_len = (matrix_dist[tsp_sequence[j], tsp_sequence[i - 1]] + matrix_dist[
            tsp_sequence[i], tsp_sequence[j + 1]])
        return - old_link_len + changed_links_len

    @staticmethod
    def loop2opt(solution, matrix_dist, max_num_of_uncrosses=10000):
        matrix_dist = np.copy(matrix_dist)
        start = t()
        new_len = compute_lenght(solution, matrix_dist)
        new_tsp_sequence = np.copy(np.array(solution))
        uncross = 0
        while uncross < max_num_of_uncrosses:
            new_tsp_sequence, new_reward, uncr_ = TwoOpt.step2opt(new_tsp_sequence, matrix_dist, new_len)
            uncross += uncr_
            if new_reward < new_len:
                new_len = new_reward
            else:
                return new_tsp_sequence.tolist(), t() - start
        return new_tsp_sequence.tolist(), t() - start



def compute_lenght(solution, dist_matrix):
    total_length = 0
    starting_node = solution[0]
    from_node = starting_node
    for node in solution[1:]:
        total_length += dist_matrix[from_node, node]
        from_node = node
    return total_length




class TwoOpt_for_GEV:
    @staticmethod
    def step2opt(solution, gev_dist, distance):
        seq_length = len(solution) - 1
        tsp_sequence = np.array(solution)
        uncrosses = 0
        for i in range(1, seq_length):
            for j in range(i + 1, seq_length):
                gain = TwoOpt_for_GEV.gain(i, j, tsp_sequence, gev_dist)
                if gain > 0:
                    uncrosses += 1
                    new_tsp_sequence = TwoOpt_for_GEV.swap2opt(tsp_sequence, i, j)
                    tsp_sequence = np.copy(new_tsp_sequence)
                    distance += gain
        return tsp_sequence, distance, uncrosses

    @staticmethod
    def swap2opt(tsp_sequence, i, j):
        new_tsp_sequence = np.copy(tsp_sequence)
        new_tsp_sequence[i:j + 1] = np.flip(tsp_sequence[i:j + 1], axis=0)  # flip or swap ?
        return new_tsp_sequence

    @staticmethod
    def gain(i, j, tsp_sequence, gev_matrix):
        old_link_len = (gev_matrix[tsp_sequence[i], tsp_sequence[i - 1]] + gev_matrix[
            tsp_sequence[j], tsp_sequence[j + 1]])
        changed_links_len = (gev_matrix[tsp_sequence[j], tsp_sequence[i - 1]] + gev_matrix[
            tsp_sequence[i], tsp_sequence[j + 1]])
        return - old_link_len + changed_links_len

    @staticmethod
    def loop2opt(solution, gev_matrix, max_num_of_uncrosses=10000):
        gev_matrix = np.copy(gev_matrix)
        start = t()
        new_len = TwoOpt_for_GEV.compute_total_gev(solution, gev_matrix)
        new_tsp_sequence = np.copy(np.array(solution))
        uncross = 0
        while uncross < max_num_of_uncrosses:
            new_tsp_sequence, new_reward, uncr_ = TwoOpt_for_GEV.step2opt(new_tsp_sequence, gev_matrix, new_len)
            uncross += uncr_
            if new_reward > new_len:
                new_len = new_reward
            else:
                return new_tsp_sequence.tolist(), t() - start

        return new_tsp_sequence.tolist(), t() - start


    @staticmethod
    def compute_total_gev(solution, gev_matrix):
        total_length = 0
        starting_node = solution[0]
        from_node = starting_node
        for node in solution[1:]:
            total_length += gev_matrix[from_node, node]
            from_node = node
        return total_length