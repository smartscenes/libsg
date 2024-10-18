"""
main.py
---
Evaluation script for running models and computing metrics on scenes.

The expected prompt CSV format can be found in the evaluation README.
"""

import json
import os
import re
import time
import traceback

import hydra
import logging
import pandas as pd
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from libsg.scene import Scene
from libsg.evaluation.utils import GPUMemoryMonitor, render_view
from libsg.scene_parser import SceneParser
from libsg.scene_builder import SceneBuilder
from libsg.scene_types import SceneSpec, SceneType, SceneGraph


load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt=r"%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOG_LEVEL", "INFO"),
)
SCENE_TEMPLATE = "scene_prompt_{}_iter_{}.json"
SCENE_GRAPH_TEMPLATE = "scene_graph_prompt_{}_iter_{}.json"
performance = {"time": 0.0, "num_generated": 0}


def compute_metrics(cfg: DictConfig, parser: SceneParser, layout_builder: SceneBuilder, prompt_data: pd.DataFrame):
    """Compute metrics to evaluate metrics across multiple prompts, i.e. each evaluated scene is a generated scene 
    conditioned on some prompt (usu. different for each prompt).

    :param cfg: config
    :param parser: parser module for extracting a scene representation (e.g. scene graph) from scenes
    :param layout_builder: layout module for for generating a scene layout from a scene prompt representation
    :param prompt_data: dataframe of prompts to evaluate
    :return: dict of the form
        {
            "success_rate": proportion of prompts for which a scene was successfully generated,
            "num_scenes": number of scenes which script attempted to generate,
            "<metric_name>": log output of metric aggregated across successfully generated scenes
        }
    """
    fail_count = 0
    total_count = 0
    metrics = []
    output = {}
    for metric in cfg.metrics:
        metrics.append(instantiate(metric))

    if not metrics:
        return

    for prompt in prompt_data[~prompt_data["test_diversity"]].itertuples():
        scene_spec = SceneSpec(SceneType.text, input=prompt.prompt, format="STK", raw=prompt.prompt)

        for iter in range(int(prompt.num_iterations)):
            total_count += 1
            scene_file = os.path.join(cfg.output, SCENE_TEMPLATE.format(prompt[0], iter))
            scene_graph_file = os.path.join(cfg.output, SCENE_GRAPH_TEMPLATE.format(prompt[0], iter))

            # if exists, reload scenes from file
            if os.path.exists(scene_file) and cfg.use_existing:
                logging.debug(f"Loading scene for prompt {prompt[0]}, iter {iter} from {scene_file}")
                with open(scene_file, "r") as f:
                    scene_state = json.load(f)
                scene = Scene.from_json(scene_state["scene"])

                if os.path.exists(scene_graph_file):
                    logging.debug(f"Loading scene graph for prompt {prompt[0]}, iter {iter} from {scene_graph_file}")
                    with open(scene_graph_file, "r") as f:
                        scene_graph_state = json.load(f)
                    scene_graph = SceneGraph.from_json(scene_graph_state)
                else:
                    scene_graph = None

            else:
                logging.debug(f"Generating scene for prompt {prompt[0]}, iter {iter}")
                try:
                    # TODO: optimize the model loading for this
                    start_time = time.time()
                    parsed = parser.parse(scene_spec)
                    scene_state = layout_builder.generate(parsed)
                    scene_graph = SceneGraph.from_json(parsed.scene_graph) if parsed.scene_graph is not None else None
                    scene = Scene.from_json(scene_state["scene"])
                    performance["time"] += time.time() - start_time
                    performance["num_generated"] += 1

                    # save scene and scene graph
                    with open(scene_file, "w") as f:
                        json.dump(scene_state, f, indent=2)
                    if parsed.scene_graph is not None:
                        with open(scene_graph_file, "w") as f:
                            json.dump(parsed.scene_graph, f, indent=2)
                except Exception as e:
                    if cfg.verbose:
                        traceback.print_exc()
                    fail_count += 1
                    continue

            for metric in metrics:
                metric(prompt[1], scene_graph, scene, prompt_data=prompt_data)

    # print metrics
    print(
        f"proportion of successfully generated scenes: {1 - fail_count / total_count:.2f} "
        f"({total_count - fail_count}/{total_count})"
    )
    output["success_rate"] = (1 - fail_count) / total_count if total_count else 0.0
    output["num_scenes"] = total_count
    for metric in metrics:
        output[metric.__class__.__name__] = metric.log()

    return output


def compute_diversity_metrics(
    cfg: DictConfig, parser: SceneParser, layout_builder: SceneBuilder, prompt_data: pd.DataFrame
):
    """Compute metrics to evaluate the diversity of multiple generations for a single prompt. Each set of diversity 
    metrics is separate per prompt.

    :param cfg: config
    :param parser: parser module for extracting a scene representation (e.g. scene graph) from scenes
    :param layout_builder: layout module for for generating a scene layout from a scene prompt representation
    :param prompt_data: dataframe of prompts to evaluate
    :return: dict of the form
        {
            "success_rate": {
                <prompt>: proportion of scenes which were successfully generated
            <metric_name>: {
                <prompt>: log output of metric aggregated across successfully generated scenes
            }
        }
    """

    diversity_metrics = []
    output = {}
    for metric in cfg.diversity_metrics:
        diversity_metrics.append(instantiate(metric))

    if not diversity_metrics:
        return

    for prompt in prompt_data[prompt_data["test_diversity"]].itertuples():
        fail_count = 0
        total_count = 0
        scene_spec = SceneSpec(SceneType.text, input=prompt.prompt, format="STK", raw=prompt.prompt)

        # reset metric states
        for metric in diversity_metrics:
            metric.reset()

        for iter in range(int(prompt.num_iterations)):
            total_count += 1
            scene_file = os.path.join(cfg.output, SCENE_TEMPLATE.format(prompt[0], iter))
            scene_graph_file = os.path.join(cfg.output, SCENE_GRAPH_TEMPLATE.format(prompt[0], iter))

            # if exists, reload scenes from file
            if os.path.exists(scene_file) and cfg.use_existing:
                logging.debug(f"Loading scene for prompt {prompt[0]}, iter {iter} from {scene_file}")
                with open(scene_file, "r") as f:
                    scene_state = json.load(f)
                scene = Scene.from_json(scene_state["scene"])

                if os.path.exists(scene_graph_file):
                    logging.debug(f"Loading scene graph for prompt {prompt[0]}, iter {iter} from {scene_graph_file}")
                    with open(scene_graph_file, "r") as f:
                        scene_graph_state = json.load(f)
                    scene_graph = SceneGraph.from_json(scene_graph_state)
                else:
                    scene_graph = None

            else:
                try:
                    # TODO: optimize the model loading for this
                    start_time = time.time()
                    parsed = parser.parse(scene_spec)
                    scene_state = layout_builder.generate(parsed)
                    scene_graph = SceneGraph.from_json(parsed.scene_graph) if parsed.scene_graph is not None else None
                    scene = Scene.from_json(scene_state["scene"])
                    performance["time"] += time.time() - start_time
                    performance["num_generated"] += 1

                    # save scene and scene graph
                    with open(scene_file, "w") as f:
                        json.dump(scene_state, f, indent=2)
                    if parsed.scene_graph is not None:
                        with open(scene_graph_file, "w") as f:
                            json.dump(parsed.scene_graph, f, indent=2)

                except Exception as e:
                    if cfg.verbose:
                        traceback.print_exc()
                    fail_count += 1
                    continue

            for metric in diversity_metrics:
                metric(prompt[1], scene_graph, scene)

        print(f"Prompt: {prompt[1]}")
        print(
            f"proportion of successfully generated scenes: {(1 - fail_count / total_count) if total_count else 0.0:.2f} "
            f"({total_count - fail_count}/{total_count})"
        )
        if "success_rate" not in output:
            output["success_rate"] = {}
        output["success_rate"][prompt.prompt] = (1 - fail_count) / total_count if total_count else 0.0
        for metric in diversity_metrics:
            if metric.__class__.__name__ not in output:
                output[metric.__class__.__name__] = {}
            output[metric.__class__.__name__][prompt.prompt] = metric.log()

    return output


@hydra.main(version_base=None, config_path="../../conf", config_name="evaluation")
def main(cfg: DictConfig):
    # 0. setup environment
    os.makedirs(cfg.output, exist_ok=True)
    gpu_monitor = GPUMemoryMonitor(interval=cfg.gpu_monitor_interval)
    gpu_monitor.start()

    # 1. Load model
    parser = SceneParser(cfg.parser, **cfg.methods)
    layout_builder = SceneBuilder(cfg.scene_builder, cfg.arch, cfg.layout, **cfg.methods)

    # 2. Load data
    prompt_data = pd.read_csv(cfg.data, quotechar='"', sep=',')
    prompt_data = prompt_data.replace({r'\n': '', r'\r': ''}, regex=True)
    
    # 3. Generate scenes from input prompts
    output = {"__info__": dict(cfg.methods)}
    output |= compute_metrics(cfg, parser, layout_builder, prompt_data)

    # 4. Generate diversity metrics per scene output
    # The main difference here is that we are calculating metrics per prompt across many trials, vs. calculating metrics
    # across many prompts.
    output |= compute_diversity_metrics(cfg, parser, layout_builder, prompt_data)

    if performance["num_generated"] > 0:
        print(f"Number of generated scenes: {performance['num_generated']}")
        print(f"Runtime per scene: {performance['time'] / performance['num_generated']:.2f}s")
        output["runtime"] = performance["time"] / performance["num_generated"]

    gpu_monitor.stop()
    max_memory_usage = gpu_monitor.join()
    print(f"Max GPU memory usage ({torch.cuda.device_count()}): {max_memory_usage} GiB")
    output["memory_gib"] = max_memory_usage

    if cfg.render_scenes:
        for scene_file in os.listdir(cfg.output):
            if re.match(r"^scene_prompt_\d+_iter\d+\.json$", scene_file):
                scene_stk_path = os.path.join(cfg.output, scene_file)
                render_output = os.path.join(cfg.output, f"{os.path.splitext(scene_file)[0]}.png")
                render_view(
                    scene_stk_path,
                    render_output,
                    semantic_rendering=cfg.semantic_index if cfg.use_semantic_render else None,
                )

    with open(cfg.metrics_output, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
