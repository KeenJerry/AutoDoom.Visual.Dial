import ntpath
import os
from pathlib import Path
from enum import Enum

_ROOT_DIR = Path(os.path.curdir).parent.parent
DATASET_TYPE = Enum("DATA_TYPE", ("train", "test"))


def _get_data_root() -> str:
    return os.path.join(_ROOT_DIR, "data_source")


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
