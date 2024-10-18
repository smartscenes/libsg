"""
generate_renders.py
---
This script can be used to generate in batch multiple image renders of a set of STK scenes. It was originally designed
for testing the rendering functionality due to some random factors in properly rendering assets.
"""

import os

import hydra
import logging
from dotenv import load_dotenv
from omegaconf import DictConfig

from libsg.evaluation.utils import render_view

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
