import numpy as np
from numpy import ndarray


def rotate_2d_point(point, rad) -> ndarray:
    x = point[0]
    y = point[1]

    rad_sin = np.sin(rad)
    rad_cos = np.cos(rad)

    rotated_x = x * rad_cos - y * rad_sin
    rotated_y = x * rad_sin + y * rad_cos

    return np.array([rotated_x, rotated_y])
