import numpy as np


class TwoOpt_CL:

    @staticmethod
    def step2opt(solution, matrix_dist, distance, cand_list, N):
        seq_length = len(solution)
        tsp_sequence = np.array(solution)
        uncrosses = 0
        for i in range(1, seq_length):
            for j_ in cand_list[i]:
                j = np.argwhere(tsp_sequence == j_)[0][0]
                jp = j + 1
                if j == N- 1:
                    jp = 0
                # print(j, j_, N)
                new_distance = distance + TwoOpt_CL.gain(i - 1, j, jp, tsp_sequence, matrix_dist)
                if new_distance < distance:
                    uncrosses += 1
                    new_tsp_sequence = TwoOpt_CL.swap2opt(tsp_sequence, i - 1, jp)
                    tsp_sequence = np.copy(new_tsp_sequence)
                    distance = new_distance

        return tsp_sequence, distance, uncrosses

    @staticmethod
    def swap2opt(tsp_sequence, i, jp):
        new_tsp_sequence = np.copy(tsp_sequence)
        if jp > i:
            new_tsp_sequence[i:jp] = np.flip(tsp_sequence[i:jp], axis=0)
        else:
            new_tsp_sequence[jp:i] = np.flip(tsp_sequence[jp:i], axis=0)
        return new_tsp_sequence

    @staticmethod
    def gain(i, j, jp, tsp_sequence, matrix_dist):
        old_link_len = (matrix_dist[tsp_sequence[i], tsp_sequence[i - 1]] +
                        matrix_dist[tsp_sequence[j], tsp_sequence[jp]])
        changed_links_len = (matrix_dist[tsp_sequence[j], tsp_sequence[i - 1]] +
                             matrix_dist[tsp_sequence[i], tsp_sequence[jp]])
        return - old_link_len + changed_links_len

    @staticmethod
    def local_search(solution, actual_len,  matrix_dist, CL):
        new_tsp_sequence = np.copy(np.array(solution))
        N = len(solution)
        uncross = 0
        while True:
            new_tsp_sequence, new_reward, uncr_ = TwoOpt_CL.step2opt(new_tsp_sequence, matrix_dist, actual_len, CL, N)
            uncross += uncr_
            if new_reward < actual_len:
                print(actual_len, new_reward)
                actual_len = new_reward
                yield new_tsp_sequence, actual_len, 0, False
            else:
                yield new_tsp_sequence, actual_len, 1, True


def twoOpt_with_cl(solution, actual_len, matrix_dist, CL):
    for data in TwoOpt_CL.local_search(solution, actual_len, matrix_dist, CL):
        if data[3]:
            return data[0], data[1]

