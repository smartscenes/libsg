import os
import sys
from typing import Optional

from omegaconf import DictConfig, open_dict
from hydra import compose


from libsg.scene_types import SceneLayoutSpec, SceneType
from libsg.model.layout import ATISS, DiffuScene, LayoutBase

__all__ = ["ATISS", "DiffuScene", "build_model"]


def build_model(
    layout_spec: SceneLayoutSpec, model_name: str, config: DictConfig, bounds: Optional[dict] = None, text_condition: bool = False,
) -> LayoutBase:
    """Build instantiation of model given specification.

    :param layout_spec: specification of scene layout, as some scenes require different models.
    :param model_name: name of model to load. One of ATISS or DiffuScene
    :param config: configuration for scene generation, toward loading model configuration
    :param bounds: dictionary to override default bounds corresponding to model
    :return: wrapper model object around core scene layout model
    """
    config_mapping = config.config[model_name]

    if layout_spec.type == SceneType.category and config_mapping.load_by_spec:
        room_type = layout_spec.input
        model_config_path = config_mapping[room_type]

        this = sys.modules[__name__]
        model_config = compose(os.path.join("layout_generator", model_config_path))

        if bounds is not None:
            model_config.layout_generator.data.bounds.update(bounds)
        with open_dict(model_config):
            model_config.layout_generator.network.room_mask_condition = not text_condition and model_config.layout_generator.network.room_mask_condition
            model_config.layout_generator.network.text_condition = text_condition

        return getattr(this, model_name)(model_config.layout_generator)
