import json
import os
from omegaconf import DictConfig

import open_clip
from langchain_openai import ChatOpenAI

from libsg.model.arch_generator.base import BaseArchGenerator
from libsg.arch import Architecture
from libsg.scene_types import ArchSpec
from libsg.model.holodeck.materials import MaterialsDB
from libsg.model.holodeck.generation.doors import DoorGenerator
from libsg.model.holodeck.generation.rooms import FloorPlanGenerator
from libsg.model.holodeck.generation.walls import WallGenerator
from libsg.model.holodeck.generation.windows import WindowGenerator


class Holodeck(BaseArchGenerator):
    """Implementation of Holodeck for architecture generation, based on LLM planning.
    
    The internal data structure is based on the ai2thor format.
    """
    
    DEFAULT_PARAMETERS = {
        "coords2d": [0, 1],
        "defaults": {
            "Ceiling": {"depth": 0.05},
            "Floor": {"depth": 0.05},
            "Wall": {"depth": 0.1, "extraHeight": 0.035},
        },
        "front": [0, 0, 1],
        "up": [0, 1, 0],
        "scaleToMeters": 1,
        "version": "holodeck@362b8ed",
        "images": [],
        "materials": [],
        "textures": [],
    }

    def __init__(self, llm_model_name: str, clip_params: DictConfig, materials_dir: str, **kwargs):
        # initialize llm
        try:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise EnvironmentError(
                "Expected an OpenAI API key in order to use the LLMSceneParser. Please set OPENAI_API_KEY and "
                "try again."
            )
        self.llm = ChatOpenAI(
            model_name=llm_model_name,
            max_tokens=2048,
            openai_api_key=openai_api_key,
        )

        # initialize CLIP
        (
            self.clip_model,
            _,
            self.clip_preprocess,
        ) = open_clip.create_model_and_transforms(clip_params.model_name, pretrained=clip_params.pretrained)
        self.clip_tokenizer = open_clip.get_tokenizer(clip_params.model_name)

        # initialize generation
        self.floor_generator = FloorPlanGenerator(self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.llm)
        self.wall_generator = WallGenerator(self.llm)
        self.door_generator = DoorGenerator(self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.llm)
        self.window_generator = WindowGenerator(self.llm)

        # additional requirements
        single_room_requirements = "I only need one room"

        single_room = kwargs.get("sceneInference.arch.singleRoom", False)
        if single_room:
            self.additional_requirements_room = single_room_requirements
        else:
            self.additional_requirements_room = "N/A"

        self.additional_requirements_door = "N/A"
        self.additional_requirements_window = "Only one wall of each room should have windows"
        self.additional_requirements_object = "N/A"
        self.additional_requirements_ceiling = "N/A"

        self.used_assets = []  # currently not used, but specifies assets to exclude

        self.materials_db = MaterialsDB(materials_dir)

    @staticmethod
    def get_empty_scene():
        with open("libsg/model/holodeck/generation/empty_house.json", "r") as f:
            return json.load(f)

    def _empty_house(self, scene: dict) -> dict:
        scene["rooms"] = []
        scene["walls"] = []
        scene["doors"] = []
        scene["windows"] = []
        scene["objects"] = []
        scene["proceduralParameters"]["lights"] = []
        return scene

    def _generate_rooms(self, scene: dict, additional_requirements_room: str, used_assets: list = None):
        if used_assets is None:
            used_assets = []
        self.floor_generator.used_assets = used_assets
        rooms = self.floor_generator.generate_rooms(scene, additional_requirements_room)
        scene["rooms"] = rooms
        return scene

    def generate_walls(self, scene: dict):
        wall_height, walls = self.wall_generator.generate_walls(scene)
        scene["wall_height"] = wall_height
        scene["walls"] = walls
        return scene

    def generate_doors(self, scene, additional_requirements_door="N/A", used_assets=[]):
        self.door_generator.used_assets = used_assets

        # generate doors
        (
            raw_doorway_plan,
            doors,
            room_pairs,
            open_room_pairs,
        ) = self.door_generator.generate_doors(scene, additional_requirements_door)
        scene["raw_doorway_plan"] = raw_doorway_plan
        scene["doors"] = doors
        scene["room_pairs"] = room_pairs
        scene["open_room_pairs"] = open_room_pairs

        # update walls
        updated_walls, open_walls = self.wall_generator.update_walls(scene["walls"], open_room_pairs)
        scene["walls"] = updated_walls
        scene["open_walls"] = open_walls
        return scene

    def generate_windows(
        self,
        scene,
        additional_requirements_window="I want to install windows to only one wall of each room",
        used_assets=[],
    ):
        self.window_generator.used_assets = used_assets
        raw_window_plan, walls, windows = self.window_generator.generate_windows(scene, additional_requirements_window)
        scene["raw_window_plan"] = raw_window_plan
        scene["windows"] = windows
        scene["walls"] = walls
        return scene

    def change_ceiling_material(self, scene):
        first_wall_material = scene["rooms"][0]["wallMaterial"]
        scene["proceduralParameters"]["ceilingMaterial"] = first_wall_material
        return scene

    def generate(self, scene_spec: ArchSpec) -> Architecture:
        """Generate an architecture based on the given specification.

        :param scene_spec: unstructured scene specification
        :raises ValueError: scene spec type not supported for generating architecture
        :return:
        """
        # initialize scene
        scene = self.get_empty_scene()
        scene["query"] = scene_spec.prompt

        # empty house
        scene = self._empty_house(scene)

        # generate rooms
        scene = self._generate_rooms(
            scene,
            additional_requirements_room=self.additional_requirements_room,
            used_assets=self.used_assets,
        )

        # generate walls
        scene = self.generate_walls(scene)

        # generate doors
        scene = self.generate_doors(
            scene,
            additional_requirements_door=self.additional_requirements_door,
            used_assets=self.used_assets,
        )

        # generate windows
        scene = self.generate_windows(
            scene,
            additional_requirements_window=self.additional_requirements_window,
            used_assets=self.used_assets,
        )
        # change ceiling material
        scene = self.change_ceiling_material(scene)

        # TODO: change format
        # TODO: fix axis alignment

        arch = self.prepare_outputs(scene)
        # arch = Architecture.from_ai2thor(scene, self.materials_db)
        return arch

    def prepare_outputs(self, scene: dict) -> Architecture:
        def mirror_x(point):
            # Only mirror the x coordinate
            return {"x": -point["x"], "y": point["y"], "z": point["z"]}

        def mirror_x_2d(point):
            # For 2D points, only mirror the x coordinate
            return [-point[0], point[1]]

        # Transform rooms
        for room in scene["rooms"]:
            room["vertices"] = [mirror_x_2d(v) for v in room["vertices"]]
            room["floorPolygon"] = [mirror_x(p) for p in room["floorPolygon"]]

        # Transform walls
        for wall in scene["walls"]:
            wall["segment"] = [mirror_x_2d(s) for s in wall["segment"]]
            if "polygon" in wall:
                wall["polygon"] = [mirror_x(p) for p in wall["polygon"]]

        # Do not need to transform doors and windows as they are relative to the wall
                
        # Convert to Architecture object
        arch = Architecture.from_ai2thor(scene, self.materials_db)
        return arch
