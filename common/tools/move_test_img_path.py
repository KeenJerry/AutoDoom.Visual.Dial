import os
import json


def move_test_img_json_path():
    test_json_path = "../../data_source/json_files/test"

    json_files = os.listdir(test_json_path)
    for file in json_files:
        content = json.loads(os.path.join(test_json_path, file))
        content["imagePath"] = content["imagePath"].replace("train", "test")


if __name__ == '__main__':
    move_test_img_json_path()
