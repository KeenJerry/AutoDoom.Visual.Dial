import numpy as np
import torch
from cv2 import cv2
from torch.utils.data import Dataset

from common.services.dataset_config_service import DataConfigService, DATASET_TYPE
from common.services.dataset_load_service import DataLoadService
from common.services.dataset_transform_service import DataTransformService, IMAGE_NET_PIXEL_MEAN, \
    IMAGE_NET_PIXEL_STD_DEVIATION


class DialButtonDataset(Dataset):
    def __init__(self, dataset_type: DATASET_TYPE, transform=None, target_transform=None):
        super(DialButtonDataset, self).__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type
        self.image_root = DataConfigService.get_image_root(dataset_type)
        self.json_file_root = DataConfigService.get_json_file_root(dataset_type)

        self.dial_keyboards = []

        if DataConfigService.has_data_cache(dataset_type):
            pass  # TODO
        else:
            self.dial_keyboards = DataLoadService.load_dial_keyboards(dataset_type)

    def _make_point_matrix(self, index):
        dial_buttons = self.dial_keyboards[index].dial_buttons
        button_number = len(dial_buttons)
        result_matrix = torch.zeros(button_number, 4, 2)
        # for i in range(button_number):
        #     result_matrix[i] =


    def __len__(self):
        return len(self.dial_keyboards)

    def __getitem__(self, index: int):
        img = cv2.imread(self.dial_keyboards[index].image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        img_width, img_height, img_channels = img.shape

        # get transform parameters
        scale, rotation, center_offset_rate, color_scale_rate = DataTransformService.random_transform_parameters()

        # get transform matrix
        transform_matrix = DataTransformService.get_transform_matrix(img_height, img_width, scale, rotation,
                                                                     center_offset_rate)

        # do affine transform
        transformed_img = DataTransformService.do_image_affine_transform(img.copy(), transform_matrix)
        if __debug__:
            cv2.imshow(img)
        # make transformed img tensor like
        # the shape should be [channels, height, width] and pixel format should be RGB not BGR
        tensor_like_img = DataTransformService.make_img_tensor_like(transformed_img)

        # apply normalization on img
        for channel in range(img_channels):
            # normalize to 0 ~ 255
            tensor_like_img[channel, :, :] = np.clip(tensor_like_img[channel, :, :] *
                                                     color_scale_rate[channel], 0, 255)
            tensor_like_img[channel, :, :] = (tensor_like_img[channel, :, :] -
                                              IMAGE_NET_PIXEL_MEAN[channel]) / IMAGE_NET_PIXEL_STD_DEVIATION[channel]

        # apply affine transform to dial button point
        dial_buttons = self.dial_keyboards[index].dial_buttons
        for i in range(len(dial_buttons)):
            dial_button = dial_buttons[i]
            dial_button.center = DataTransformService.do_point_affine_transform(dial_button.center, transform_matrix)
            dial_button.left_top_point = DataTransformService.do_point_affine_transform(
                dial_button.left_top_point, transform_matrix)
            dial_button.left_bottom_point = DataTransformService.do_point_affine_transform(
                dial_button.left_bottom_point, transform_matrix)
            dial_button.right_top_point = DataTransformService.do_point_affine_transform(
                dial_button.right_top_point, transform_matrix)
            dial_button.right_bottom_point = DataTransformService.do_point_affine_transform(
                dial_button.right_bottom_point, transform_matrix)

        # generate label for heatmap

        # get reinforced button point location for AE method
        # reinforced_points =
