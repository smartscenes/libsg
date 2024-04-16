from omegaconf import DictConfig

from libsg.scene_types import SceneSpec, SceneType
from .base import BaseSceneParser


class RoomTypeParser(BaseSceneParser):
    """
    Parse an unstructured scene description into a structured scene specification which can be used by downstream
    modules.

    The parsing function currently expects a "text" type scene specification and expects to find one of the
    following key phrases—"living room", "dining room", or "bedroom"—in the text description, which is used to
    specify the scene room type to generate. As of now, the scene parsing is fairly rudimentary and does not use
    any other information in the scene specification.
    """

    def __init__(self, cfg: DictConfig) -> None:
        pass

    def parse(self, scene_spec: SceneSpec) -> SceneSpec:
        if scene_spec.type == SceneType.text:
            if "living room" in scene_spec.input:
                return SceneSpec(
                    type=SceneType.category,
                    input="livingroom",
                    format=scene_spec.format,
                    raw=scene_spec.input,
                    room_type="livingroom",
                )
            elif "dining room" in scene_spec.input:
                return SceneSpec(
                    type=SceneType.category,
                    input="diningroom",
                    format=scene_spec.format,
                    raw=scene_spec.input,
                    room_type="diningroom",
                )
            elif "bedroom" in scene_spec.input:
                return SceneSpec(
                    type=SceneType.category,
                    input="bedroom",
                    format=scene_spec.format,
                    raw=scene_spec.input,
                    room_type="bedroom",
                )
            else:
                raise ValueError(f"Cannot parse room type from scene specification: {scene_spec.input}")
        else:
            raise ValueError(f"Cannot parse scene type from scene specification: {scene_spec.type}")
