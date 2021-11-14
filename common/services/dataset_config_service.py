import ntpath
import os
from enum import Enum

from common.services.project_config_service import ROOT_DIR

DATASET_TYPE = Enum("DATASET_TYPE", ("train", "test"))

DST_HEIGHT = 384
DST_WIDTH = 384
IMAGE_NET_PIXEL_MEAN: list[float] = [123.67500, 116.28000, 103.53000]
IMAGE_NET_PIXEL_STD_DEVIATION: list[float] = [58.39500, 57.12000, 57.37500]


def _get_data_root() -> str:
    return os.path.join(ROOT_DIR, "data_source")


class DataConfigService:
    @staticmethod
    def has_data_cache(dataset_type: DATASET_TYPE) -> bool:
        if os.path.exists(os.path.join(_get_data_root(), "cache", "cache_{}.pkl".format(dataset_type.name))):
            return True
        else:
            return False

    @staticmethod
    def get_image_root(dataset_type: DATASET_TYPE) -> ntpath:
        return os.path.join(_get_data_root(), "images", dataset_type.name)

    @staticmethod
    def get_json_file_root(dataset_type: DATASET_TYPE) -> ntpath:
        return os.path.join(_get_data_root(), "json_files", dataset_type.name)


