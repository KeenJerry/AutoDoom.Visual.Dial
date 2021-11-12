import numpy as np
from numpy import ndarray


def generate_gaussian_core(sigma: int, bound: int) -> ndarray:
    size = 2 * sigma * bound + 1
    x = np.arange(0, size, 1, dtype=float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2  # mean coordinate
    # The gaussian is not normalized, we want the center(peak) value to equal 1
    return np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
