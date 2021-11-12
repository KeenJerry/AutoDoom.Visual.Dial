import ntpath
import os

import numpy as np
from numpy import ndarray

from common.services.dataset_config_service import DataConfigService, DATASET_TYPE
from common.tools.parser import parse_json_file


class DialButton:
    def __init__(self, left_top_point: ndarray, left_bottom_point: ndarray, right_bottom_point: ndarray,
                 right_top_point: ndarray):
        self.left_top_point = left_top_point
        self.left_bottom_point = left_bottom_point
        self.right_bottom_point = right_bottom_point
        self.right_top_point = right_top_point
        self.center = np.mean(np.vstack((self.left_top_point, self.left_bottom_point, self.right_bottom_point,
                                         self.right_top_point)), axis=0)


class DialKeyBoard:
    def __init__(self, dial_buttons: list[DialButton], image_path: ntpath, image_width: int, image_height: int):
        self.dial_buttons = dial_buttons
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height


class DataLoadService:
    @staticmethod
    def load_dial_keyboards(dataset_type: DATASET_TYPE) -> list[DialKeyBoard]:
        dial_keyboards = []
        json_file_root: ntpath = DataConfigService.get_json_file_root(dataset_type)
        json_file_paths = [os.path.join(json_file_root, file) for file in os.listdir(json_file_root)
                           if os.path.isfile(os.path.join(json_file_root, file))]

        for file_path in json_file_paths:
            dial_buttons = []
            parsed_json_data = parse_json_file(file_path)

            for shape in parsed_json_data["shapes"]:
                dial_buttons.append(
                    DialButton(
                        left_top_point=np.array(shape["points"][0]),
                        left_bottom_point=np.array(shape["points"][1]),
                        right_bottom_point=np.array(shape["points"][2]),
                        right_top_point=np.array(shape["points"][3])
                    )
                )

            dial_keyboards.append(
                DialKeyBoard(
                    dial_buttons=dial_buttons,
                    image_path=os.path.join(json_file_root, parsed_json_data["imagePath"]),
                    image_width=parsed_json_data["imageWidth"],
                    image_height=parsed_json_data["imageHeight"]
                )
            )

        return dial_keyboards
