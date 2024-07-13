import json
import logging
import os
from pprint import pprint

from omegaconf import DictConfig
from pyvis.network import Network

from libsg.scene_types import SceneSpec
from libsg.model.sg_parser import build_parser_model


class SceneParser:
    """
    Parse an unstructured scene description into a structured scene specification which can be used by downstream
    modules.

    The current implementation of this class involves very basic room parsing for the layout module, but in the future
    we would want to have more sophisticated parsing of the scene type, shape, and objects generated, such as with a
    scene graph or other representation.
    """

    NEXT_ID = 0
    OUTPUT_DIR = ".data/evaluation/output"

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        self.model_name = kwargs.get("sceneInference.parserModel", cfg.default_model)  # legacy format
        self.visualize_sg = kwargs.get("sceneInference.parser.visualize", False)
        self.cfg = cfg

    def parse(self, scene_spec: SceneSpec) -> SceneSpec:
        """Parse scene description into a structured scene specification.

        :param scene_spec: unstructured scene specification
        :raises ValueError: scene spec type or room type not supported for parsing
        :return: structured scene specification
        """
        parser_model = build_parser_model(scene_spec, self.model_name, self.cfg)
        parsed_spec = parser_model.parse(scene_spec)
        logging.debug(f"Scene Graph:")
        if logging.DEBUG >= logging.root.level:
            pprint(parsed_spec.scene_graph)

        # sg visualization
        if self.visualize_sg:
            SceneParser.visualize(parsed_spec.scene_graph)
        SceneParser.NEXT_ID += 1

        return parsed_spec

    @staticmethod
    def visualize(scene_graph):
        net = Network(directed=True)
        for obj in scene_graph["objects"]:
            net.add_node(obj["id"], label=obj["name"], color="blue")
            for attribute in obj["attributes"]:
                net.add_node(f"{obj['id']}-{attribute}", label=attribute, color="yellow")
                net.add_edge(obj["id"], f"{obj['id']}-{attribute}")

        for relationship in scene_graph["relationships"]:
            net.add_edge(relationship["subject_id"], relationship["target_id"], label=relationship["type"])

        output_path = os.path.join(SceneParser.OUTPUT_DIR, f"scene_graph_{SceneParser.NEXT_ID}.html")
        net.show(output_path, notebook=False)
