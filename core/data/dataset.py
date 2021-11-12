import numpy as np
import torch
from cv2 import cv2
from torch.utils.data import Dataset

from common.services.dataset_config_service import DataConfigService, DATASET_TYPE,\
    IMAGE_NET_PIXEL_MEAN, IMAGE_NET_PIXEL_STD_DEVIATION
from common.services.dataset_load_service import DataLoadService
from common.services.dataset_transform_service import DataTransformService


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
        self.dial_keyboards[index].do_affine_transform_on_buttons(transform_matrix)

        # generate label for heatmap
        label = self.dial_keyboards[index].generate_heatmap(sigma=2, bound=3, heatmap_width=96, heatmap_height=96)

        # get reinforced button point location for AE method
        self.dial_keyboards[index].calculate_reinforced_button_points()
        ground_truth_points = self.dial_keyboards[index].aggregate_reinforced_points()

        return tensor_like_img, label, ground_truth_points
