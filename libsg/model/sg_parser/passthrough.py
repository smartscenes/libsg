from omegaconf import DictConfig

from libsg.scene_types import SceneSpec, SceneType
from .base import BaseSceneParser


class PassThroughParser(BaseSceneParser):
    """
    Empty parser.
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        pass

    def parse(self, scene_spec: SceneSpec) -> SceneSpec:
        scene_spec.raw = scene_spec.input
        return scene_spec
