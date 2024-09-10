import json
import logging
import os
import subprocess
import threading
import time

import torch


def render_view(
    scene_stk_path: str,
    output_path: str,
    view_index: int = 4,
    asset_groups: list[str] = None,
    semantic_rendering: str = None,
):
    """
    Generate png render of a scene in STK format. Supports both "realistic" and semantic renderings.

    Please see the sstk/scene-toolkit repositories for the render-file.js script and
    https://github.com/smartscenes/sstk/wiki/Batch-rendering for more details about the batch rendering script.

    :param scene_stk_path: path to scene JSON file in STK format.
    :param output_path: path to output destination for png file.
    :param view_index: index representing default view angle in the range of [0-7] (default/left/right/bottom/top/front/back). Defaults to 4 (top-down)
    :param asset_groups: list of asset sources to use in rendering. Defaults to None, in which case the asset sources
    are directly inferred from the scene STK file.
    :param semantic_rendering: str path to index CSV file specifying colors of each object type. The CSV file should be
    formatted as follows:

    index,label,color
    0,unknown,#ffffff
    1,Floor,#d3d3d3
    2,Kids bed,#98df8a
    ...

    Defaults to None, in which case a "realistic" rendering is generated.
    """
    env = os.environ.copy()
    env["NODE_BASE_URL"] = "http://aspis.cmpt.sfu.ca/scene-toolkit"
    render_file_path = os.environ.get("RENDER_FILE_PATH")

    if render_file_path is None:
        logging.error("Could not find render-file.js filepath. 2D renderings will not be generated.")
        return

    # load asset groups
    if asset_groups:
        asset_groups_str = ",".join(asset_groups)
    else:
        with open(scene_stk_path, "r") as f:
            scene = json.load(f)
        asset_sources = scene["scene"]["assetSource"]
        texture_source = scene["scene"]["arch"]["defaults"]["textureSource"]
        asset_groups_str = ",".join([*asset_sources, texture_source])

    if semantic_rendering:
        subprocess.run(
            [
                render_file_path,
                "--assetType",
                "scene",
                "--assetGroups",
                asset_groups_str,
                "--view_index",
                str(view_index),
                "--color_by",
                "objectType",
                "--index",
                semantic_rendering,
                "--input",
                scene_stk_path,
                "--output",
                output_path,
            ],
            env=env,
        )
    else:
        subprocess.run(
            [
                render_file_path,
                "--assetType",
                "scene",
                "--assetGroups",
                asset_groups_str,
                "--view_index",
                str(view_index),
                "--envmap",
                "neutral",
                "--use_lights",
                "--input",
                scene_stk_path,
                "--output",
                output_path,
            ],
            env=env,
        )


class GPUMemoryMonitor(threading.Thread):
    def __init__(self, interval: float = 1):
        super().__init__()
        self.interval = interval
        self._stop_event = threading.Event()

        self.num_devices = torch.cuda.device_count()
        self.max_gpu_util_gb = 0

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            gpu_memory = sum(torch.cuda.max_memory_allocated(i) / 1024**3 for i in range(self.num_devices))
            self.max_gpu_util_gb = max(self.max_gpu_util_gb, gpu_memory)
            time.sleep(self.interval)

    def join(self, *args, **kwargs) -> None:
        super().join(*args, **kwargs)
        return self.max_gpu_util_gb
