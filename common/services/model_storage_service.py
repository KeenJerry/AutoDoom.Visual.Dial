import os

from common.services.project_config_service import ROOT_DIR


def _get_output_root() -> str:
    return os.path.join(ROOT_DIR, "output")


def get_model_save_path():
    return os.path.join(_get_output_root(), "model")
