import h5py
import sys, os
import numpy as np
from contextlib import contextmanager
from scipy.spatial.distance import pdist


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def create_upper_matrix(values, size):
    """
    builds an upper matrix
    @param values: to insert in the matrix
    @param size: of the matrix
    @return:
    """
    upper = np.zeros((size, size))
    r = np.arange(size)
    mask = r[:, None] < r
    upper[mask] = values
    return upper



def give_me_dist(sequence):
    max_length = sequence.shape[0]
    distance = create_upper_matrix(pdist(sequence, "euclidean"), max_length)
    distance = distance.T + distance
    return distance



def shape_for_h5(x):
    x_to_save = np.stack(x.nonzero(), axis=1).astype(np.uint8)
    return x_to_save


class pixel_in_image:
    def __init__(self, max_value, num_pixel, raggio):
        self.max_value = max_value
        self.num_pixel = num_pixel
        self.raggio = raggio

    def pixel_pos(self, vec):
        num_pixel, raggio = map(float, [self.num_pixel,  self.raggio])
        max_value = float(self.max_value)
        x = int(round((vec[0]) / max_value * (num_pixel // 2 - 2 * raggio)) + num_pixel // 2)
        y_ = round((vec[1]) / max_value * (num_pixel // 2 - 2 * raggio))
        y = int(-y_ + (num_pixel // 2))
        return int(x), y

def is_dataset(h5f):
    return isinstance(h5f, h5py.Dataset)

def is_group(h5f):
    return isinstance(h5f, h5py.Group)

def from_ind_to_dense(xy,  settings):
    x, y = xy
    number_channels = 1
    if settings.with_MST:
        number_channels = 3
    num_pixel = settings.num_pixels
    x_tensor = np.zeros((num_pixel, num_pixel, number_channels))
    for i, j, k in zip(x[:, 0], x[:, 1], x[:, 2]):
        x_tensor[i, j, k] = 1.

    y_tensor = np.zeros((num_pixel, num_pixel, 1))
    for i, j, k in zip(y[:, 0], y[:, 1], y[:, 2]):
        y_tensor[i, j, k] = 1.
    return x_tensor, y_tensor


def transformation(pos, operation):
    pos = np.copy(pos)
    pos -= np.mean(pos, axis=0)
    pos = (pos - pos.min()) / (pos.max() - pos.min())
    pos -= np.array([0.5, 0.5])
    max_value = np.max(np.linalg.norm(pos, axis=1))
    return pos, max_value