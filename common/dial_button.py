import numpy as np
from numpy import ndarray

from common.services.dataset_transform_service import DataTransformService


def _get_heatmap_overlap(point, gaussian_radius, heatmap_width, heatmap_height):
    mu_x = int(point[0] / 4 + 0.5)
    mu_y = int(point[1] / 4 + 0.5)

    ul = [int(mu_x - gaussian_radius), int(mu_y - gaussian_radius)]  # ul:upper left, br:bottom right
    br = [int(mu_x + gaussian_radius) + 1, int(mu_y + gaussian_radius + 1)]

    if ul[0] >= heatmap_width or ul[1] >= heatmap_height or br[0] < 0 or br[1] < 0:
        return None, None, None, None

    g_x = max(0, -ul[0]), min(br[0], heatmap_width) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heatmap_height) - ul[1]

    img_x = max(0, ul[0]), min(br[0], heatmap_width)
    img_y = max(0, ul[1]), min(br[1], heatmap_height)

    return g_x, g_y, img_x, img_y


class DialButton:
    def __init__(self, left_top_point: ndarray, left_bottom_point: ndarray, right_bottom_point: ndarray,
                 right_top_point: ndarray):
        self.left_top_point = left_top_point
        self.left_bottom_point = left_bottom_point
        self.right_bottom_point = right_bottom_point
        self.right_top_point = right_top_point
        self.center = np.mean(np.vstack((self.left_top_point, self.left_bottom_point, self.right_bottom_point,
                                         self.right_top_point)), axis=0)

        self.reinforced_left_top_point = np.zeros(3)
        self.reinforced_left_bottom_point = np.zeros(3)
        self.reinforced_right_bottom_point = np.zeros(3)
        self.reinforced_right_top_point = np.zeros(3)

    def do_affine_transform(self, transform_matrix):
        self.center = DataTransformService.do_point_affine_transform(
            self.center, transform_matrix)
        self.left_top_point = DataTransformService.do_point_affine_transform(
            self.left_top_point, transform_matrix)
        self.left_bottom_point = DataTransformService.do_point_affine_transform(
            self.left_bottom_point, transform_matrix)
        self.right_top_point = DataTransformService.do_point_affine_transform(
            self.right_top_point, transform_matrix)
        self.right_bottom_point = DataTransformService.do_point_affine_transform(
            self.right_bottom_point, transform_matrix)

    def calculate_reinforced_points(self):
        temp_left_bottom_point = (self.left_bottom_point / 4 + 0.5).astype(int)
        temp_left_top_point = (self.left_top_point / 4 + 0.5).astype(int)
        temp_right_top_point = (self.right_top_point / 4 + 0.5).astype(int)
        temp_right_bottom_point = (self.right_bottom_point / 4 + 0.5).astype(int)

        self.reinforced_left_bottom_point = np.array([0 * 96 * 96 + temp_left_bottom_point[1] * 96 +
                                                      temp_left_bottom_point[0], 1,
                                                      temp_left_bottom_point[0] + temp_left_bottom_point[1]])

        self.reinforced_left_top_point = np.array([0 * 96 * 96 + temp_left_top_point[1] * 96 +
                                                   temp_left_top_point[0], 1,
                                                   temp_left_top_point[0] + temp_left_top_point[1]])

        self.reinforced_right_top_point = np.array([0 * 96 * 96 + temp_right_top_point[1] * 96 +
                                                    temp_right_top_point[0], 1,
                                                    temp_right_top_point[0] + temp_right_top_point[1]])

        self.reinforced_right_bottom_point = np.array([0 * 96 * 96 + temp_right_bottom_point[1] * 96 +
                                                       temp_right_bottom_point[0], 1,
                                                       temp_right_bottom_point[0] + temp_right_bottom_point[1]])

    def draw_heatmap(self, gaussian_core, heatmap_width, heatmap_height, label):
        gaussian_radius = int(gaussian_core.shape[0] / 2)
        g_x, g_y, img_x, img_y = _get_heatmap_overlap(self.left_top_point, gaussian_radius, heatmap_width,
                                                      heatmap_height)
        if g_x is not None:
            label[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(gaussian_core[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                                                                        label[0][img_y[0]:img_y[1], img_x[0]:img_x[1]])

        g_x, g_y, img_x, img_y = _get_heatmap_overlap(self.left_bottom_point, gaussian_radius, heatmap_width,
                                                      heatmap_height)
        if g_x is not None:
            label[1][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(gaussian_core[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                                                                        label[1][img_y[0]:img_y[1], img_x[0]:img_x[1]])

        g_x, g_y, img_x, img_y = _get_heatmap_overlap(self.right_bottom_point, gaussian_radius, heatmap_width,
                                                      heatmap_height)
        if g_x is not None:
            label[2][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(gaussian_core[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                                                                        label[2][img_y[0]:img_y[1], img_x[0]:img_x[1]])

        g_x, g_y, img_x, img_y = _get_heatmap_overlap(self.right_top_point, gaussian_radius, heatmap_width,
                                                      heatmap_height)
        if g_x is not None:
            label[3][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(gaussian_core[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                                                                        label[3][img_y[0]:img_y[1], img_x[0]:img_x[1]])

        g_x, g_y, img_x, img_y = _get_heatmap_overlap(self.center, gaussian_radius, heatmap_width,
                                                      heatmap_height)
        if g_x is not None:
            label[4][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(gaussian_core[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                                                                        label[4][img_y[0]:img_y[1], img_x[0]:img_x[1]])

    def aggregate_reinforced_points(self):
        return [
            self.reinforced_left_top_point,
            self.reinforced_left_bottom_point,
            self.reinforced_right_bottom_point,
            self.reinforced_right_top_point
        ]
