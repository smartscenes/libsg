from pprint import pprint

from omegaconf import DictConfig

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

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        self.model_name = kwargs.get("sceneInference.parserModel", cfg.default_model)
        self.cfg = cfg

    def parse(self, scene_spec: SceneSpec) -> SceneSpec:
        """Parse scene description into a structured scene specification.

        :param scene_spec: unstructured scene specification
        :raises ValueError: scene spec type or room type not supported for parsing
        :return: structured scene specification
        """
        parser_model = build_parser_model(scene_spec, self.model_name, self.cfg)
        parsed_spec = parser_model.parse(scene_spec)
        print("Scene Graph:")
        pprint(parsed_spec.scene_graph)

        return parsed_spec
