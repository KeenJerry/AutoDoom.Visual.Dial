import numpy as np
from numpy import ndarray
from cv2 import cv2

from common.services.dataset_config_service import DST_HEIGHT, DST_WIDTH
from common.tools.transform import rotate_2d_point


class DataTransformService:
    @staticmethod
    def random_transform_parameters() -> (float, float, float, ndarray, list[float]):
        scale: float = np.clip(np.random.randn(), -0.5, 1.0) * 0.25 + 1
        rotation: float = np.clip(np.random.randn(), -1.0, 1.0) * 15 if np.random.randn() < 0.6 else 0
        color_scale_rate = [np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2)]
        center_offset_rate = np.abs(np.clip(np.random.randn(2), -1.0, 1.0)) * 0.1

        return scale, rotation, center_offset_rate, color_scale_rate

    @staticmethod
    def get_transform_matrix(src_height: int, src_width: int, scale: float,
                             rotation: float, center_offset_rate: ndarray):

        src_points = np.zeros(3, 2)
        dst_points = np.zeros(3, 2)

        # calculate src_points
        rotation_rad = rotation * np.pi / 180.0

        src_point1 = np.zeros(1, 2)
        src_point2 = rotate_2d_point(np.array([0, src_height]), rotation_rad)
        src_point3 = rotate_2d_point(np.array([src_width, 0]), rotation_rad)

        src_points[0, :] = src_point1
        src_points[1, :] = src_point2
        src_points[2, :] = src_point3

        # calculate dst_points
        if src_height >= src_width:
            actual_width = DST_HEIGHT * src_width / src_height
            actual_height = DST_HEIGHT
        else:
            actual_width = DST_WIDTH
            actual_height = DST_WIDTH * src_height / src_width

        center_offset = center_offset_rate * np.array([DST_WIDTH, DST_HEIGHT])

        dst_point1 = np.zeros(2) + center_offset
        dst_point2 = dst_point1 + np.array([0, actual_height * scale])
        dst_point3 = dst_point1 + np.array([actual_width * scale, 0])

        dst_points[0, :] = dst_point1
        dst_points[1, :] = dst_point2
        dst_points[2, :] = dst_point3

        return cv2.getAffineTransform(src_points, dst_points)

    @staticmethod
    def do_image_affine_transform(img, transform_matrix):
        return cv2.warpAffine(img, transform_matrix, (DST_WIDTH, DST_HEIGHT), flags=cv2.INTER_CUBIC)

    @staticmethod
    def make_img_tensor_like(img):
        tensor_like_image = img.copy()
        tensor_like_image = np.transpose(tensor_like_image, (2, 0, 1))

        tensor_like_image = tensor_like_image[::-1, :, :]
        return tensor_like_image.astype(np.float)

    @staticmethod
    def do_point_affine_transform(point, transform_matrix):
        return np.dot(transform_matrix, np.array([point[0], point[1], 1.0]).T)[0: 2]
