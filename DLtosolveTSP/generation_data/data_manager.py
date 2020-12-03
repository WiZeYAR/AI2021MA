import h5py
import torch
import cv2 as cv
import numpy as np
import os, tempfile
import multiprocessing as mp
from concorde.tsp import TSPSolver
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist

from DLtosolveTSP.tools.dir_manager import Dir_Manager
from DLtosolveTSP.generation_data.utils import create_upper_matrix, pixel_in_image, \
     shape_for_h5, from_ind_to_dense, plot_cv, transformation


class Generate_Images:

    def __init__(self, settings, if_save=False, dir_file=None):
        self.settings = settings
        self.num_pixels = settings.num_pixels

    def __call__(self):
        number_cities = np.random.randint(20, 100)
        pos = np.random.uniform(-0.5, 0.5, size=number_cities * 2).reshape((number_cities, 2))
        return number_cities, pos

    @staticmethod
    def distance_mat(pos, number_cities):
        distance = create_upper_matrix(pdist(pos, "euclidean"), number_cities)
        distance = np.round((distance.T + distance)*1000, 0)/1000
        return distance

    def create_instance(self, j):
        np.random.seed(j)
        num_cities, pos = self.__call__()
        return num_cities, pos

    def create_data(self, j, test=False):
        np.random.seed(j)
        num_cities, pos = self.create_instance(j)
        dir = os.getcwd()
        with tempfile.TemporaryDirectory() as path:
            os.chdir(path)
            solver = TSPSolver.from_data(
                pos[:, 0] * 1000,
                pos[:, 1] * 1000,
                norm="EUC_2D"
            )
            solution = solver.solve()

        os.chdir(dir)

        if test:
            dist_mat = Generate_Images.distance_mat(pos, num_cities)
            input_image, output_image = self.create_images(self.settings, num_cities, pos, dist_mat, solution.tour)
            return num_cities, pos, dist_mat, input_image, output_image, solution.tour
        else:
            return num_cities, pos, solution.tour


    def create_instances(self, num_instances, seed_starting_from, parallel=True):
        data = [self.create_data(j+seed_starting_from) for j in range(num_instances)]
        return self.organize_data(data)

    @staticmethod
    def organize_data(data):
        all_dimensions, all_pos, all_tours = ([] for _ in range(3))
        for i in range(len(data)):
            all_dimensions.append(data[i][0])
            all_pos.append(data[i][1])
            all_tours.append(data[i][2])

        all_dimensions, all_pos, all_tours = map(np.array,(all_dimensions, all_pos,
                                                           all_tours))
        return all_dimensions, all_pos, all_tours

    def save(self, data, seed, hf, num_inst_file):
        all_dimensions, all_pos, all_tours = data
        for it in range(num_inst_file):
            seed_to_add = seed + it
            group = hf.create_group(f'seed_{seed_to_add}')
            group.create_dataset(f"num_cities", shape=(1,),
                                 dtype=np.int, chunks=True, data=np.array(all_dimensions[it]))

            group.create_dataset(f"pos", shape=all_pos[it].shape,
                                 dtype=np.float, chunks=True, data=all_pos[it])


            group.create_dataset(f"optimal_tour", shape=all_tours[it].shape,
                                 dtype=np.int, chunks=True,
                                 data=all_tours[it])

    @staticmethod
    def create_images(settings, num_cities, pos, dist_mat, tour):
        raggio_nodo = settings.ray_dot
        spess_edge = settings.thickness_edge
        num_pixels = settings.num_pixels

        rosso = (1, 0, 0)
        pos, max_value = transformation(pos, operation= settings.operation)
        pixel_man = pixel_in_image(max_value=max_value, num_pixel=num_pixels, raggio=raggio_nodo)

        input_im = np.zeros((num_pixels, num_pixels, 1), np.uint8)
        for i in range(pos.shape[0]):
            c_x, c_y = pixel_man.pixel_pos(vec=pos[i])
            cv.circle(input_im, (c_x, c_y), raggio_nodo, rosso,
                      thickness=-1, lineType=cv.LINE_AA)


        out_im = np.zeros((num_pixels, num_pixels, 1), np.uint8)
        for i,j_ in zip(tour, np.roll(tour, 1)):
            f_x, f_y = pixel_man.pixel_pos(vec=pos[i])
            t_x, t_y = pixel_man.pixel_pos(vec=pos[j_])
            out_im = cv.line(out_im, (f_x, f_y), (t_x, t_y), rosso, spess_edge,
                             lineType=cv.LINE_AA)

        return input_im, out_im



class TrainDatasetHandler(Dataset):
    def __init__(self, settings, path_files, epoch):
        self.settings = settings
        self.path = path_files
        self.files_saved = sort_the_list_of_files(self.path)
        self.len = len(self.files_saved) * settings.num_instances_x_file
        self.starting_seed = 10000
        self.train = False
        self.file = False
        self.epoch = epoch
        self.angolo = 45 * self.epoch

    def __len__(self):
        return self.len

    def load_file(self):
        self.file = h5py.File(f"{self.path}/{self.files_saved[self.index_file]}", "r")


    def __getitem__(self, slice):

        if self.chek_if_to_load(slice):
            self.index_file = self.from_slice_to_indexfile(slice)
            self.load_file()

        all_input, all_output = [], []

        index_slice = [self.starting_seed + j for j in slice_iterator(slice)]
        for key in index_slice:
            number_cities = self.file[f'//seed_{key}'][f'num_cities'][...]
            pos = self.file[f'//seed_{key}'][f'pos'][...]
            tour = self.file[f'//seed_{key}'][f'optimal_tour'][...]
            number_cities = number_cities[0]
            dist_mat = Generate_Images.distance_mat(pos, number_cities)
            input_image, output_image = Generate_Images.create_images(self.settings, number_cities, pos,
                                                                      dist_mat, tour)

            input_images = np.transpose(input_image, [2, 0, 1])
            output_images = np.transpose(output_image, [2, 0, 1])
            all_input.append(input_images)
            all_output.append(output_images)
            self.seed = key

        if self.train == True:
            X = torch.tensor(all_input , dtype=torch.float, requires_grad=False)
            Y = torch.tensor(all_output, dtype=torch.float, requires_grad=False)
            return X, Y

        else:
            return map(np.array, (all_input, all_output))


    def chek_if_to_load(self, slice):
        if str(slice).isdigit():
            if int(slice) % self.settings.num_instances_x_file == 0:
                return True
            else:
                if self.file:
                    return False
                else:
                    return True
        else:
            start_sl = str(slice).replace("(", ",").replace(")", ",").split(',')[1]
            if start_sl == 'None':
                return True
            else:
                if int(start_sl) % self.settings.num_instances_x_file == 0:
                    return True
                else:
                    if self.file:
                        return False
                    else:
                        return True

    def from_slice_to_indexfile(self, slice):
        if str(slice).isdigit():
            return int(slice) // self.settings.num_instances_x_file
        start_sl = str(slice).replace("(", ",").replace(")", ",").split(',')[1]
        if start_sl == 'None':
            return 0
        else:
            return int(start_sl) // self.settings.num_instances_x_file


# USEFUL FUNCTIONS
def sort_the_list_of_files(path):
    list_files = os.listdir(path)
    dic_files = {int(f_n[:-8]): f_n for f_n in list_files}
    sorted_list_of_file = [el[1] for el in sorted(dic_files.items(), key=lambda kv: kv[0])]
    return sorted_list_of_file


def slice_iterator(slice):
    if not str(slice).isdigit():
        start_sl, stop_sl, step_sl = str(slice).replace("(", ",").replace(")", ",").split(',')[1:4]
        # print(start_sl, stop_sl, step_sl)
        # print(len(start_sl), len(stop_sl), len(step_sl))
        if start_sl == 'None':
            iterator_slice = range(int(stop_sl))
            if step_sl != ' None':
                assert False, "step of the iterator should be 0"
        else:
            iterator_slice = 1
            if stop_sl != "None":
                iterator_slice = range(int(start_sl), int(stop_sl))
                if step_sl != " None":
                    assert False, "step of the iterator should be 0"
    else:
        iterator_slice = [slice]

    return iterator_slice




class Generate_Data_for_Test:

    def __init__(self, settings):
        self.settings = settings
        self.num_pixels = settings.num_pixels
        self.dir_ent = Dir_Manager(settings)

    def __call__(self, number_cities, seed_):
        np.random.seed(seed_)
        pos = np.random.uniform(-0.5, 0.5, size=number_cities* 2).reshape((number_cities, 2))
        distance = create_upper_matrix(pdist(pos, "euclidean"), number_cities)
        distance = distance.T + distance
        return pos, distance

    def create_data(self, test_dimensions,  number_tests):
        seed_ = 0
        for number_cities in test_dimensions:
            tour_array = np.load(self.dir_ent.path_to_risultati(number_cities, "optimal_tours"))
            for j in range(number_tests):
                pos, dist_mat = self.__call__(number_cities, seed_)
                seed_ += 1
                input_image, output_image = Generate_Images.create_images(self.settings, number_cities, pos,
                                                                          dist_mat, tour_array[j])

                yield number_cities, pos, dist_mat, input_image, tour_array[j], j, output_image


    def create_solutions(self,test_dimensions,  number_tests):
        seed_ = 0
        for number_cities in test_dimensions:
            list_to_save = []
            for _ in range(number_tests):
                pos, dist_mat = self.__call__(number_cities, seed_)
                seed_ += 1
                dir = os.getcwd()
                with tempfile.TemporaryDirectory() as path:
                    os.chdir(path)
                    solver = TSPSolver.from_data(
                        pos[:, 0] * 1000,
                        pos[:, 1] * 1000,
                        norm="EUC_2D"
                    )
                    solution = solver.solve()
                os.chdir(dir)
                list_to_save.append(solution.tour)

            np.save(self.dir_ent.path_to_risultati(number_cities, "optimal_tours"), np.array(list_to_save))

        print("all test results saved.")

