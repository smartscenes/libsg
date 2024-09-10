from omegaconf import DictConfig
from hydra.utils import instantiate

from .base import BaseArchGenerator
from .square import SquareRoomGenerator
from libsg.scene_types import ArchSpec


__all__ = ["SquareRoomGenerator"]


def build_arch_model(arch_spec: ArchSpec, model_name: str, config: DictConfig, **kwargs) -> BaseArchGenerator:
    """Build instantiation of scene description parser model given specification.

    :param scene_spec: specification of scene, as some scenes require different models.
    :param model_name: name of model to load. One of "LLM", "RoomType", or "InstructScene"
    :param config: configuration for model to return
    :return: wrapper model object around core scene parser model
    """
    model_config = config.config[model_name]
    return instantiate(model_config)
