import traceback

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from libsg.evaluation import metrics as metrics_methods
from libsg.scene import Scene
from libsg.scene_parser import SceneParser
from libsg.scene_builder import SceneBuilder
from libsg.scene_types import SceneSpec, SceneType, SceneGraph


def compute_metrics(cfg: DictConfig, parser: SceneParser, layout_builder: SceneBuilder, prompt_data: pd.DataFrame):
    fail_count = 0
    total_count = 0
    metrics = []
    for metric in cfg.metrics:
        metrics.append(instantiate(metric))

    if not metrics:
        return

    for prompt in prompt_data[~prompt_data["test_diversity"]].itertuples():
        scene_spec = SceneSpec(SceneType.text, input=prompt[1], format="STK", raw=prompt[1])

        for _ in range(int(prompt[2])):
            try:
                # TODO: optimize the model loading for this
                parsed = parser.parse(scene_spec)
                scene_state = layout_builder.generate(parsed)
                scene_graph = SceneGraph.from_json(parsed.scene_graph) if parsed.scene_graph is not None else None
                scene = Scene.from_json(scene_state["scene"])
            except Exception as e:
                if cfg.verbose:
                    traceback.print_exc()
                fail_count += 1
            else:
                for metric in metrics:
                    metric(prompt[1], scene_graph, scene)
            total_count += 1

    # print metrics
    print(
        f"proportion of successful scenes: {1 - fail_count / total_count:.2f} "
        f"({total_count - fail_count}/{total_count})"
    )
    for metric in metrics:
        metric.log()


def compute_diversity_metrics(
    cfg: DictConfig, parser: SceneParser, layout_builder: SceneBuilder, prompt_data: pd.DataFrame
):
    diversity_metrics = []
    for metric in cfg.diversity_metrics:
        diversity_metrics.append(instantiate(metric))

    if not diversity_metrics:
        return

    for prompt in prompt_data[prompt_data["test_diversity"]].itertuples():
        fail_count = 0
        total_count = 0
        scene_spec = SceneSpec(SceneType.text, input=prompt[1], format="STK", raw=prompt[1])

        # reset metric states
        for metric in diversity_metrics:
            metric.reset()

        for _ in range(int(prompt[2])):
            try:
                # TODO: optimize the model loading for this
                parsed = parser.parse(scene_spec)
                scene_state = layout_builder.generate(parsed)
                scene_graph = SceneGraph.from_json(parsed.scene_graph) if parsed.scene_graph is not None else None
                scene = Scene.from_json(scene_state["scene"])
            except Exception as e:
                if cfg.verbose:
                    traceback.print_exc()
                fail_count += 1
            else:
                for metric in diversity_metrics:
                    metric(prompt[1], scene_graph, scene)
            total_count += 1

        print(f"Prompt: {prompt[1]}")
        print(
            f"proportion of successful scenes: {1 - fail_count / total_count:.2f} "
            f"({total_count - fail_count}/{total_count})"
        )
        for metric in diversity_metrics:
            metric.log()


@hydra.main(version_base=None, config_path="../../conf", config_name="evaluation")
def main(cfg: DictConfig):
    # 1. Load model
    parser = SceneParser(cfg.parser, **{"sceneInference.parserModel": cfg.parser_method})
    layout_builder = SceneBuilder(cfg.scene_builder, cfg.layout, **{"sceneInference.layoutModel": cfg.layout_method})

    # 2. Load data
    prompt_data = pd.read_csv(cfg.data)

    # 3. Generate scenes from input prompts
    compute_metrics(cfg, parser, layout_builder, prompt_data)

    # 4. Generate diversity metrics per scene output
    # The main difference here is that we are calculating metrics per prompt across many trials, vs. calculating metrics
    # across many prompts.
    compute_diversity_metrics(cfg, parser, layout_builder, prompt_data)


if __name__ == "__main__":
    main()
