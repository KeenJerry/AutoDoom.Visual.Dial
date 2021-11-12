import ntpath
import os

import numpy as np

from common.dial_button import DialButton
from common.dial_keyboard import DialKeyboard
from common.services.dataset_config_service import DataConfigService, DATASET_TYPE
from common.tools.parser import parse_json_file


class DataLoadService:
    @staticmethod
    def load_dial_keyboards(dataset_type: DATASET_TYPE) -> list[DialKeyboard]:
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
                DialKeyboard(
                    dial_buttons=dial_buttons,
                    image_path=os.path.join(json_file_root, parsed_json_data["imagePath"]),
                    image_width=parsed_json_data["imageWidth"],
                    image_height=parsed_json_data["imageHeight"]
                )
            )

        return dial_keyboards
