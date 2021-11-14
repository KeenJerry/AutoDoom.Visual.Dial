import ntpath

import numpy as np
from numpy import ndarray

from common.dial_button import DialButton
from common.tools.gaussian import generate_gaussian_core


class DialKeyboard:
    def __init__(self, dial_buttons: list[DialButton], image_path: ntpath, image_width: int, image_height: int):
        self.dial_buttons = dial_buttons
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height

    def do_affine_transform_on_buttons(self, transform_matrix):
        for i in range(len(self.dial_buttons)):
            self.dial_buttons[i].do_affine_transform(transform_matrix)

    def calculate_reinforced_button_points(self):
        for i in range(len(self.dial_buttons)):
            self.dial_buttons[i].calculate_reinforced_points()

    def generate_heatmap(self, sigma, bound, heatmap_width, heatmap_height) -> ndarray:
        label_channel_number = 5
        label = np.zeros((label_channel_number, heatmap_width, heatmap_height), np.float)
        gaussian_core = generate_gaussian_core(sigma, bound)
        for i in range(len(self.dial_buttons)):
            self.dial_buttons[i].draw_heatmap(gaussian_core, heatmap_width, heatmap_height, label)
        return label

    def aggregate_reinforced_points(self) -> ndarray:
        ground_truth_points = np.zeros((120, 4, 3))
        for i in range(len(self.dial_buttons)):
            ground_truth_points[i] = self.dial_buttons[i].aggregate_reinforced_points()
        return ground_truth_points
