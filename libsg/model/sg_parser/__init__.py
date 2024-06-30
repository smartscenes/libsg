import os
import sys

from omegaconf import DictConfig, OmegaConf
from hydra import compose

from .base import BaseSceneParser
from .llm_sg_parser import LLMSceneParser as LLM
from .room_type import RoomTypeParser as RoomType
from .instructscene import InstructSceneParser as InstructScene
from libsg.scene_types import SceneSpec


__all__ = ["LLM", "RoomType", "InstructScene"]


def build_parser_model(scene_spec: SceneSpec, model_name: str, config: DictConfig) -> BaseSceneParser:
    """Build instantiation of scene description parser model given specification.

    :param scene_spec: specification of scene, as some scenes require different models.
    :param model_name: name of model to load. One of "LLM", "RoomType", or "InstructScene"
    :param config: configuration for model to return
    :return: wrapper model object around core scene parser model
    """
    config_mapping = config.config[model_name]

    if config_mapping.load_by_spec:
        room_type = RoomType(OmegaConf.create({})).parse(scene_spec).input

        model_config_path = config_mapping[room_type]
    else:
        model_config_path = config_mapping.get("config")

    this = sys.modules[__name__]
    if model_config_path:
        model_config = compose(os.path.join("scene_parser", model_config_path))
    else:
        model_config = OmegaConf.create({"scene_parser": {}})

    return getattr(this, model_name)(model_config.scene_parser)
