import os
from pathlib import Path
from absl import logging

PATH_GLI_STR = os.environ.get("GLI_PATH")
if PATH_GLI_STR is None:
    raise ValueError("GLI_PATH environment variable not set.")

PATH_GLI = Path(PATH_GLI_STR)
if not PATH_GLI.exists():
    raise ValueError("GLI_PATH does not exist.")

PATH_STORAGE = PATH_GLI / "storage"

PATH_ASSET = PATH_STORAGE / "assets"

WANDB_PATH_LOG = PATH_STORAGE / "logs"


def get_path_log(tag):
    return WANDB_PATH_LOG / tag

