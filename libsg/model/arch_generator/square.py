from libsg.arch import Architecture
from libsg.scene_types import ArchSpec
from .base import BaseArchGenerator


class SquareRoomGenerator(BaseArchGenerator):
    """
    Parse an unstructured scene description into a structured scene specification which can be used by downstream
    modules.

    The parsing function currently expects a "text" type scene specification and expects to find one of the
    following key phrases—"living room", "dining room", or "bedroom"—in the text description, which is used to
    specify the scene room type to generate. As of now, the scene parsing is fairly rudimentary and does not use
    any other information in the scene specification.
    """

    DEFAULTS = {
        "Wall": {"depth": 0.1, "extraHeight": 0.035},
        "Ceiling": {"depth": 0.05},
        "Ground": {"depth": 0.08},
        "Floor": {"depth": 0.05},
        "textureSource": "smtTexture",
    }
    DEFAULT_MATERIAL = [{"name": "surface", "diffuse": "#ffffff", "texture": "wood_cream_plane_1375"}]

    def __init__(self, room_size: list[float], version: str = "arch@1.0.2", **kwargs) -> None:
        self.room_size = room_size  # [x, z]
        self.version = version
        self.up = kwargs.get("up", [0, 0, 1])
        self.front = kwargs.get("front", [0, 1, 0])
        self.scale_to_meters = kwargs.get("scale_to_meters", 1.0)

    def generate(self, arch_spec: ArchSpec) -> Architecture:
        # default initializations
        arch = Architecture(id="generated-arch")
        arch.version = self.version
        arch.up = self.up
        arch.front = self.front
        arch.defaults = self.DEFAULTS
        arch.unit = self.scale_to_meters

        arch.add_element(
            {
                "id": 0,
                "type": "Floor",
                "roomId": "0",
                "points": [
                    [self.room_size[0] / 2, self.room_size[1] / 2, 0],
                    [-self.room_size[0] / 2, self.room_size[1] / 2, 0],
                    [-self.room_size[0] / 2, -self.room_size[1] / 2, 0],
                    [self.room_size[0] / 2, -self.room_size[1] / 2, 0],
                ],
                "offset": [0, 0, 0],
                "materials": self.DEFAULT_MATERIAL,
            }
        )
        return arch
