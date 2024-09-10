import json
import os
import subprocess
import time
import traceback

import hydra
import logging
import pandas as pd
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from libsg.evaluation import metrics as metrics_methods
from libsg.evaluation.utils import render_view
from libsg.scene import Scene
from libsg.scene_parser import SceneParser
from libsg.scene_builder import SceneBuilder
from libsg.scene_types import SceneSpec, SceneType, SceneGraph, ArchSpec

load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt=r"%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOG_LEVEL", "INFO"),
)


@hydra.main(version_base=None, config_path="../../conf", config_name="inference")
def main(cfg: DictConfig):
    if not hasattr(cfg, "input") or not hasattr(cfg, "output"):
        raise ValueError(
            "Input and output must be specified in the command line, e.g. "
            "python generate_renders.py +input=... +output=..."
        )

    for i in range(getattr(cfg, "num_renders", 1)):
        render_view(
            cfg.input,
            os.path.join(cfg.output, f"scene_render_{i}.png"),
            semantic_rendering=cfg.semantic_index if cfg.use_semantic_render else None,
        )


if __name__ == "__main__":
    main()
