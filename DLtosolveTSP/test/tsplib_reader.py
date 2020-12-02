import numpy as np
from DLtosolveTSP.tools.dir_manager import Dir_Manager
from DLtosolveTSP.generation_data.data_manager import Generate_Images


class read_tspfiles:
    def __init__(self):
        self.path = "./problems/TSP/"
        self.files = ["eil51.tsp", "berlin52.tsp", "st70.tsp",
                      "eil76.tsp", "pr76.tsp", "rat99.tsp",
                      "kroA100.tsp", "kroC100.tsp", "rd100.tsp",
                      "eil101.tsp", "lin105.tsp", "bier127.tsp",
                      "ch130.tsp", "kroA150.tsp", "kroA200.tsp"]

    def instances_generator(self):
        for file in self.files[:]:
            yield self.read_instance(self.path + file)

    def read_instance(self, name_tsp):
        # read raw data
        file_object = open(name_tsp)
        data = file_object.read()
        file_object.close()
        self.lines = data.splitlines()

        # store data set information
        self.name = self.lines[0].split(' ')[1]
        self.nPoints = np.int(self.lines[3].split(' ')[1])

        # read all data points and store them
        self.pos = np.zeros((self.nPoints, 2))
        for i in range(self.nPoints):
            line_i = self.lines[6 + i].split(' ')
            self.pos[i, 0] = float(line_i[1])
            self.pos[i, 1] = float(line_i[2])

        self.create_dist_matrix()
        return self.nPoints, self.pos, self.dist_matrix, self.name

    @staticmethod
    def distance_euc(zi, zj):
        delta_x = zi[0] - zj[0]
        delta_y = zi[1] - zj[1]
        return round(np.sqrt((delta_x) ** 2 + (delta_y) ** 2), 0)

    def create_dist_matrix(self):
        self.dist_matrix = np.zeros((self.nPoints, self.nPoints))

        for i in range(self.nPoints):
            for j in range(i, self.nPoints):
                self.dist_matrix[i, j] = self.distance_euc(self.pos[i], self.pos[j])
        self.dist_matrix += self.dist_matrix.T



class Generate_Data_for_TSPLib_files:

    def __init__(self, settings):
        self.settings = settings
        self.num_pixels = settings.num_pixels
        self.dir_ent = Dir_Manager(settings)
        self.reader = read_tspfiles()


    def create_data(self):
        for number_cities, pos, dist_mat, name in self.reader.instances_generator():
            tour = optimal_solver(name)
            input_image, output_image = Generate_Images.create_images(self.settings, number_cities, pos, dist_mat, tour)
            yield number_cities, pos, dist_mat, input_image, name, output_image



def optimal_solver(name_instance):
    return np.load(f"./data/tsplib_problems/optimal/{name_instance}.npy")
