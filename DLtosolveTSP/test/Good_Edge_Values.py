import math
import torch
import cv2 as cv
import numpy as np

from DLtosolveTSP.tools.dir_manager import Dir_Manager
from DLtosolveTSP.network.CNN_for_TSP import CNN_for_GoodEdgeDistribution
from DLtosolveTSP.generation_data.utils import pixel_in_image, transformation


class GoodEdgeValues:
    def __init__(self, settings):
        self.settings = settings
        self.dir_ent = Dir_Manager(settings)
        self.device = torch.device('cuda')
        self.model = CNN_for_GoodEdgeDistribution()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.dir_ent.final_model_file,
                                              map_location=self.device))


    def create_GEV(self, data):
        if self.settings.operation == "test":
            num_cities, self.pos, self.dist_mat, input_image, tour, number, real_image = data
            name = str(number)
        else:
            num_cities, self.pos, self.dist_mat, input_image, name, real_image = data
        input_image = np.transpose(input_image, [2, 0, 1 ])
        X = torch.tensor(np.expand_dims(input_image, 0), dtype=torch.float,
                         requires_grad=True, device=self.device)

        criterion = torch.nn.MSELoss(reduction='sum')
        self.model.eval()
        out_image = self.model(X)
        real_for_torch = torch.tensor(np.expand_dims(np.transpose(real_image,
                                                                  [2, 0, 1]), 0),
                                      dtype=torch.float, requires_grad=False, device=self.device)
        loss = criterion(out_image, real_for_torch)
        self.loss = loss.detach().cpu().numpy()
        out_image_np = np.transpose(out_image.detach().cpu().numpy()[0], [1, 2, 0])

        self.gev = self.calc_edgevalue(num_cities, self.pos, out_image_np, X)


    def calc_edgevalue(self, n, pos, out_image_p, X):
        pos = np.copy(pos)
        out_image = np.squeeze(out_image_p)
        out_image = (out_image - np.min(out_image)) / (np.max(out_image) - np.min(out_image))

        pos, max_value = transformation(pos, operation=self.settings.operation)

        raggio_nodo = self.settings.ray_dot
        edge_width = self.settings.thickness_edge
        pixel_man = pixel_in_image(max_value=max_value,
                                   num_pixel=out_image.shape[0],
                                   raggio=raggio_nodo)

        rho = rho_fun(raggio_nodo + 0.5, edge_width/ 2.0)
        img_size = (out_image.shape[0], out_image.shape[1])
        N = np.zeros(img_size, dtype=np.uint8)

        for i in range(pos.shape[0]):
            c_x, c_y = pixel_man.pixel_pos(vec=pos[i])
            cv.circle(N, (c_x, c_y), raggio_nodo, 255, -1)

        ve = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            # image of all nodes except node i
            c_x, c_y = pixel_man.pixel_pos(vec=pos[i])
            punto_i = cv.circle(np.zeros(img_size, dtype=np.uint8),
                                (c_x, c_y), raggio_nodo, 255,
                                thickness=-1)
            N_i = N - punto_i

            for j in range(i + 1, n):
                # image of edge (i, j)
                c_xj, c_yj = pixel_man.pixel_pos(vec=pos[j])


                punto_j = cv.circle(np.zeros(img_size, dtype=np.uint8),
                                    (c_xj, c_yj), raggio_nodo, 255,
                                    thickness= -1)

                edge = cv.line(np.zeros(img_size, dtype=np.uint8),
                               (c_x, c_y), (c_xj, c_yj), 1,
                               edge_width,
                               lineType=cv.LINE_AA).astype(np.float32)

                N_ij = np.maximum(0, N_i - punto_j)
                tau = (N_ij.astype(np.float32) / 255.0 /rho * edge).sum()


                if np.sum(punto_i * (punto_j/255.)) > 1000 :
                    ve[i, j] = ve[j, i] = 1.1

                ve[i, j] = ve[j, i] = (out_image * edge).sum() / (1.0 + tau)\
                                      / (edge.sum() + 1e-10)


        return np.maximum(ve, 0)



rho_fun = (lambda r, d:
    2 * (d * math.sqrt(r ** 2 - d ** 2) + r ** 2 * math.asin(d / r))
    if d < r else
    2 * math.pi * r ** 2
    )
