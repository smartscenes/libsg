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
    # 1. setup environment
    methods = cfg.methods
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load data
    prompt_data = pd.read_csv(cfg.data)

    # 2b. Unfold prompt data structure
    processed_prompts = []
    for prompt in prompt_data.itertuples():
        for i in range(prompt.num_iterations):
            processed_prompts.append(
                {
                    "index": len(processed_prompts),
                    "prompt_id": prompt[0],
                    "iter": i,
                    "description": prompt.prompt,
                    "rendered": {},
                }
            )

    # 3. generate scenes and render
    arch_ids = {}
    runtimes = {}
    for method in methods:
        total_time = 0.0
        num_generated = 0
        os.makedirs(os.path.join(output_dir, method.name), exist_ok=True)
        parser = SceneParser(cfg.parser, **method.params)
        layout_builder = SceneBuilder(cfg.scene_builder, cfg.arch, cfg.layout, **method.params)

        index = 0
        for prompt in prompt_data.itertuples():
            for i in range(prompt.num_iterations):
                arch_scene_id, room_ids = arch_ids.get(index) if arch_ids.get(index) else (None, None)

                scene_spec = SceneSpec(
                    SceneType.text,
                    input=prompt.prompt,
                    format="STK",
                    raw=prompt.prompt,
                    arch_spec=ArchSpec(type="id", input=arch_scene_id, format="STK", room_ids=room_ids),
                )
                scene_stk_path = os.path.join(output_dir, method.name, f"scene_prompt_{index}.json")
                if cfg.skip_existing_stk and os.path.exists(scene_stk_path):
                    with open(scene_stk_path, "r") as f:
                        scene_state = json.load(f)
                else:
                    try:
                        # TODO: optimize the model loading for this
                        start_time = time.time()
                        parsed = parser.parse(scene_spec)
                        scene_state = layout_builder.generate(parsed)

                    except Exception as e:
                        logging.error(
                            f"Failed to generate scene for prompt ({prompt[0]}) on iteration {i} using method {method.name}."
                        )
                        traceback.print_exc()
                        scene_state = None
                        processed_prompts[index]["rendered"][method.name] = False
                    else:
                        total_time += time.time() - start_time
                        num_generated += 1
                        with open(scene_stk_path, "w") as f:
                            json.dump(scene_state, f, indent=2)

                # cache arch id
                if index not in arch_ids:
                    arch_ids[index] = (
                        scene_state["scene"]["arch"]["id"],
                        [region["id"] for region in scene_state["scene"]["arch"]["regions"]],
                    )

                render_output = os.path.join(output_dir, method.name, f"scene_prompt_{index}.png")
                if scene_state is not None and not (cfg.skip_existing_render and os.path.exists(render_output)):
                    render_view(
                        scene_stk_path,
                        render_output,
                        semantic_rendering=cfg.semantic_index if cfg.use_semantic_render else None,
                    )

                processed_prompts[index]["rendered"][method.name] = os.path.exists(render_output)

                index += 1

        runtimes[method.name] = total_time / num_generated if num_generated > 0 else None

    with open(os.path.join(output_dir, cfg.output_prompt_file), "w") as f:
        json.dump({"data": processed_prompts, "runtimes": runtimes}, f, indent=2)


if __name__ == "__main__":
    main()
