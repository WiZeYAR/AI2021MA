import numpy as np



class TwoOpt_CL:

    @staticmethod
    def step2opt(solution, matrix_dist, distance, cand_list):
        seq_length = len(solution)
        tsp_sequence = np.array(solution)
        uncrosses = 0
        for i in range(1, seq_length):
            for j in cand_list[i]:
                new_tsp_sequence = TwoOpt_CL.swap2opt(tsp_sequence, i - 1, j)
                new_distance = distance + TwoOpt_CL.gain(i - 1, j, tsp_sequence, matrix_dist)
                if new_distance < distance:
                    uncrosses += 1
                    tsp_sequence = np.copy(new_tsp_sequence)
                    distance = new_distance

        return tsp_sequence, distance, uncrosses

    @staticmethod
    def swap2opt(tsp_sequence, i, j):
        new_tsp_sequence = np.copy(tsp_sequence)
        final_index = j + 1
        new_tsp_sequence[i:final_index] = np.flip(tsp_sequence[i:final_index], axis=0)
        return new_tsp_sequence

    @staticmethod
    def gain(i, j, tsp_sequence, matrix_dist):
        old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[i - 1]] + matrix_dist[
            tsp_sequence[j], tsp_sequence[j + 1]])
        changed_links_len = (matrix_dist[tsp_sequence[j], tsp_sequence[i - 1]] + matrix_dist[
            tsp_sequence[i], tsp_sequence[j + 1]])
        return - old_link_len + changed_links_len

    @staticmethod
    def local_search(solution, actual_len,  matrix_dist, CL):
        new_tsp_sequence = np.copy(np.array(solution))
        uncross = 0
        while True:
            new_tsp_sequence, new_reward, uncr_ = TwoOpt_CL.step2opt(new_tsp_sequence, matrix_dist, actual_len, CL)
            uncross += uncr_
            if new_reward < actual_len:
                actual_len = new_reward
                yield new_tsp_sequence, actual_len, 0, False
            else:
                yield new_tsp_sequence, actual_len, 1, True





def twoOpt(solution, actual_len, matrix_dist, CL):
    for data in TwoOpt_CL.local_search(solution, actual_len, matrix_dist, CL):
        if data[3]:
            return data[0], data[1]
