from omegaconf import DictConfig
from hydra.utils import instantiate

from .base import BaseArchGenerator
from .holodeck import Holodeck
from .square import SquareRoomGenerator
from libsg.scene_types import ArchSpec


__all__ = ["SquareRoomGenerator", "Holodeck"]


def build_arch_model(arch_spec: ArchSpec, model_name: str, config: DictConfig, **kwargs) -> BaseArchGenerator:
    """Build instance of architecture generation model given specification.

    :param arch_spec: specification of architecture, in circumstance that model instantiation depends on which arch is
    being generated.
    :param model_name: name of model to load. One of "SquareRoomGenerator" or "Holodeck"
    :param config: configuration for model to return
    :return: model instance for architecture generation
    """
    model_config = config.config[model_name]
    return instantiate(model_config)
