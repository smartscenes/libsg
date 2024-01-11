"""
scene_builder.py
---
Main code for generating a scene from a given scene description. This code currently handles the entire overall
pipeline, including architecture generation/retrieval, layout generation, and object generation/retrieval. In the 
future, some of the functionality here may be split off into other classes.
"""

import json
import numpy as np
import sys
from easydict import EasyDict
from omegaconf import DictConfig
from typing import Any, Optional

import numpy as np

from libsg.assets import AssetDb
from libsg.object_placement import ObjectPlacer
from libsg.scene_types import (
    Arch,
    BBox3D,
    Ceiling,
    Door,
    Floor,
    JSONDict,
    SceneState,
    ObjectSpec,
    ObjectTemplateInstance,
    Opening,
    Point3D,
    Room,
    SceneSpec,
    SceneLayoutSpec,
    SceneLayout,
    PlacementSpec,
    Wall,
    Window,
)
from libsg.scene import Scene
from libsg.simscene import SimScene
from libsg.simulator import Simulator
from libsg.geo import Transform
from libsg.io import SceneExporter
from libsg.model import build_model

COLORS = [
    [31, 119, 180],
    [152, 223, 138],
    [140, 86, 75],
    [199, 199, 199],
    [174, 199, 232],
    [214, 39, 40],
    [196, 156, 148],
    [188, 189, 34],
    [255, 127, 14],
    [255, 152, 150],
    [227, 119, 194],
    [219, 219, 141],
    [255, 187, 120],
    [148, 103, 189],
    [247, 182, 210],
    [23, 190, 207],
    [44, 160, 44],
    [197, 176, 213],
    [127, 127, 127],
    [158, 218, 229],
]


class SceneBuilder:
    """Generate scene based on prompt"""

    NEAR_FLOOR_HEIGHT = 0.05

    def __init__(self, cfg: DictConfig, layout: DictConfig):
        """
        :param cfg: loaded config at conf/config.yaml:scene_builder
        :param layout: loaded config at conf/config.yaml:scene_builder.layout
        """
        self.layout_cfg = layout
        self.__base_solr_url = cfg.get("solr_url")
        self.__arch_db = AssetDb("arch", cfg.get("arch_db"))
        self.__scene_db = AssetDb("scene", cfg.get("scene_db"))
        self.__model_db = AssetDb("model", cfg.get("model_db"), solr_url=f"{self.__base_solr_url}/models3d")
        self.scene_exporter = SceneExporter()
        self.object_placer = ObjectPlacer(model_db=self.__model_db)

    def generate_arch(self, scene_spec: SceneSpec) -> tuple[Arch, JSONDict, BBox3D]:
        """Generate architecture for scene.

        The current implementation retrieves an architecture from an existing HSSD scene and removes all objects,
        preserving only walls (and associated openings) and the floor shape.

        :param scene_spec: specification of scene. As of current, assumes that scene spec specifies a pre-existing
        HSSD scene ID to load as the base scene, though all objects are removed from the architecture.
        :raises ValueError: unsupported element or hole type
        :return: tuple of architecture object, scene object (without objects), and axis-aligned bounding box around
        scene, as calculated by the min/max AABB of all elements in the scene.
        """
        # TODO: move this code to arch_builder.py
        # load base scene and clear default layout objects
        base_scene_path = self.__scene_db.get(scene_spec.input)
        base_scene = json.load(open(base_scene_path, "r"))["scene"]
        base_scene["object"] = []  # keep arch, drop existing objects

        # compute base scene AABB to transform object positions later
        with Simulator(mode="direct", verbose=False, use_y_up=False) as sim:
            sim_scene = SimScene(sim, Scene.from_json(base_scene), self.__model_db.config, include_ground=True)
            base_scene_aabb = sim_scene.aabb

        # create architecture
        scene_arch = base_scene["arch"]
        elements = []
        rooms: dict[str, Room] = {}  # room_id -> Room
        openings: list[Opening] = []
        for elem_raw in scene_arch["elements"]:
            room_id = elem_raw["roomId"]

            if room_id not in rooms:
                rooms[room_id] = Room(room_id)

            # TODO: use from_json() for these instead of constructing from scratch
            if elem_raw["type"] == "Wall":
                elem = Wall(
                    elem_raw["id"],
                    elem_raw["points"],
                    elem_raw["height"],
                    elem_raw["depth"],
                    elem_raw["materials"],
                    room_id,
                )
                rooms[room_id].wall_sides.append(elem)

                for hole in elem_raw.get("holes", []):
                    opening_min = hole["box"]["min"]
                    opening_max = hole["box"]["max"]
                    width = opening_max[0] - opening_min[0]
                    mid = (opening_min[0] + (width / 2)) / elem.width
                    height = opening_max[1] - opening_min[1]
                    elevation = opening_min[1] + height / 2
                    if hole["type"] == "window":
                        openings.append(
                            Window(
                                id=hole["id"],
                                parent=elem,
                                mid=mid,
                                height=height,
                                width=width,
                                elevation=elevation,
                            )
                        )
                    elif hole["type"] == "door":
                        openings.append(
                            Door(
                                id=hole["id"],
                                parent=elem,
                                mid=mid,
                                height=height,
                                width=width,
                                elevation=elevation,
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported hole type: {hole['type']}")

            elif elem_raw["type"] == "Floor":
                elem = Floor(
                    elem_raw["id"],
                    [Point3D(*p) for p in elem_raw["points"]],
                    elem_raw.get("depth", scene_arch["defaults"]["Floor"]["depth"]),
                    elem_raw["materials"],
                    room_id,
                )
                rooms[room_id].floor = elem

            elif elem_raw["type"] == "Ceiling":
                elem = Ceiling(
                    elem_raw["id"],
                    [Point3D(*p) for p in elem_raw["points"]],
                    elem_raw.get("depth", scene_arch["defaults"]["Floor"]["depth"]),
                    elem_raw["materials"],
                    room_id,
                )
                rooms[room_id].ceiling = elem
            else:
                raise ValueError(f"Unsupported type: {elem['type']}")
            elements.append(elem)

        arch = Arch(elements=elements, rooms=list(rooms.values()))

        return arch, base_scene, base_scene_aabb

    def generate_layout(
        self, layout_spec: SceneLayoutSpec, base_scene_aabb: BBox3D, model_name: Optional[str] = None
    ) -> SceneLayout:
        """Generate coarse scene layout given specification of architecture and room type.

        :param layout_spec: specification for architecture and room type
        :param model_name: name of model. If not provided, the default model is pulled from the configuration.
        :return: scene layout, including architecture and coarse object types and poses
        """
        if model_name is None:
            model_name = self.layout_cfg.default_model

        # can set bounds based on scene itself, but the problem is that the object proximities are not really scaled
        # appropriately, so either some objects will be out of scene or a lot of objects end up overlapping (or both)
        # TODO: bounds need to be relative to object centroids rather than full extent of scene
        base_scene_aabb_norm = BBox3D(
            base_scene_aabb.min - base_scene_aabb.centroid, base_scene_aabb.max - base_scene_aabb.centroid
        )
        bounds = {
            "translations": [
                [base_scene_aabb_norm.min.x, base_scene_aabb_norm.min.y, base_scene_aabb_norm.min.z],
                [base_scene_aabb_norm.max.x, base_scene_aabb_norm.max.y, base_scene_aabb_norm.max.z],
            ]
        }
        layout_model = build_model(layout_spec, model_name, self.layout_cfg, bounds=None)
        objects = layout_model.generate(layout_spec)
        print(objects)

        layout = SceneLayout(
            objects=[
                ObjectTemplateInstance(
                    label=obj["wnsynsetkey"],
                    dimensions=obj["dimensions"],
                    position=obj["position"],
                    orientation=obj["orientation"],
                )
                for obj in objects
            ],
            arch=layout_spec.arch,
        )
        if self.layout_cfg.export_layout:
            layout.export_coarse_layout("coarse_layout.obj")
        return layout

    def generate_objects(self, scene_state: JSONDict, base_scene_centroid: Point3D, scene_layout: SceneLayout) -> Scene:
        """Generate object based on scene state, with placement relative to base scene centroid

        :param scene_state: base scene state including architecture only
        :param base_scene_centroid: point representing centroid of scene (i.e. midpoint of axis-aligned bounding box of
        scene)
        :param scene_layout: description of coarse scene layout for objects
        :return: scene state with merged architecture and retrieved/generated objects
        """
        # create ObjectSpec and PlacementSpec for each object and place in scene
        for obj in scene_layout.objects:
            synset = obj.label
            dimensions = obj.dimensions
            position = Point3D.add(
                Point3D.fromlist(obj.position), base_scene_centroid
            )  # TODO heuristically re-center to base scene centroid
            position.z = self.NEAR_FLOOR_HEIGHT  # TODO Heuristically shift to near-floor placement
            orientation = obj.orientation - np.pi / 2  # TODO figure out why theta adjustment seems to be required here
            if "pendant_lamp" in synset or "ceiling_lamp" in synset:
                continue  # skip ceiling lamps  # TODO: ceiling lamp placement logic
            object_spec = {"type": "category", "wnsynsetkey": synset, "dimensions": dimensions}
            placement_spec = {
                "type": "placement_point",
                "position": position,
                "orientation": orientation,
                "allow_collisions": True,
            }
            scene_state = self.object_add(scene_state, object_spec, placement_spec)

        return Scene.from_json(scene_state)

    def export_scene(self, scene: Scene, format: str) -> JSONDict:
        """Export scene to specified format for downstream rendering.

        :param scene: scene object
        :param format: format for exporting. Currently either HAB (habitat) or STK (scene toolkit) formats are
        supported.
        :raises ValueError: unsupported scene format
        :return: JSON representing scene in specified format
        """
        if format == "HAB":
            # Habitat object instance pre-rotation
            preRquat = Transform.get_alignment_quaternion([0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, -1])
            preR = Transform()
            preR.set_rotation(preRquat)
            for m in scene.modelInstances:
                m.transform.rightMultiply(preR)

            # export scene
            scene_json = self.scene_exporter.export(scene, format=SceneExporter.SceneFormat.HAB)
        elif format == "STK":
            scene_json = self.scene_exporter.export(scene, format=SceneExporter.SceneFormat.STK)
        else:
            raise ValueError(f"Scene format not supported: {format}")

        return scene_json

    def generate(self, scene_spec: SceneSpec) -> JSONDict:
        """Generate a scene based on scene specification.

        :param scene_spec: Specification of scene to generate
        :return: JSON in specified format representing scene
        """
        # generate or retrieve architecture
        # TODO: use input scene_spec instead of an arbitrary arch_spec
        arch_spec = SceneSpec(type="id", input="102344115", format=scene_spec.format)
        arch, scene_state, base_scene_aabb = self.generate_arch(arch_spec)
        base_scene_centroid = base_scene_aabb.centroid

        # generate layout
        scene_layout_spec = SceneLayoutSpec(type=scene_spec.type, input=scene_spec.input, arch=arch)
        scene_layout = self.generate_layout(scene_layout_spec, base_scene_aabb)

        # generate or retrieve objects
        scene = self.generate_objects(scene_state, base_scene_centroid, scene_layout)

        # format scene
        return self.export_scene(scene, scene_spec.format)

    def modify(self, scene_state: JSONDict, description: str) -> JSONDict:
        raise NotImplementedError

    def retrieve(self, scene_spec: SceneSpec) -> JSONDict:
        """Retrieve scene based on scene specification.

        :param scene_spec: specification for scene to retrieve
        :return: JSON of retrieved scene from database
        """
        if scene_spec.type == "id":
            scenestate_path = self.__scene_db.get(scene_spec.input)
            scenestate = json.load(open(scenestate_path, "r"))
            return scenestate
        else:
            raise ValueError(f"Scene specification type not supported for retrieval: {scene_spec.type}")

    def object_remove(self, scene_state: SceneState, object_spec: ObjectSpec) -> JSONDict:
        """Remove existing object from given scene.

        :param scene_state: state of existing scene
        :param object_spec: specification describing object to be removed
        :return: JSON of new scene state after object removal
        """
        scene = Scene.from_json(scene_state)
        object_spec = object_spec if isinstance(object_spec, EasyDict) else EasyDict(object_spec)

        removed = []
        if object_spec.type == "object_id":
            id_to_remove = object_spec.object
            removed_obj = scene.remove_object_by_id(id_to_remove)
            if removed_obj is not None:
                removed.append(removed_obj)
        elif object_spec.type == "model_id":
            model_id_to_remove = object_spec.object
            removed = scene.remove_objects(lambda obj: obj.model_id == model_id_to_remove)
        elif object_spec.type == "category":
            fq = self.object_placer.object_query_constraints(object_spec)
            results = self.__model_db.search(object_spec.object, fl="fullId", fq=fq, rows=25000)
            model_ids = set([result["fullId"] for result in results])
            removed = scene.remove_objects(lambda obj: obj.model_id in model_ids)
        else:
            print(f"Unsupported object_spec.type={object_spec.type}", file=sys.stderr)

        scene.modifications.extend([{"type": "removed", "object": r.to_json()} for r in removed])
        return scene.to_json()

    def object_add(self, scene_state: JSONDict, object_spec: ObjectSpec, placement_spec: PlacementSpec) -> JSONDict:
        """Add new object to existing scene.

        :param scene_state: state of existing scene
        :param object_spec: specification describing object to be added
        :param placement_spec: specification describing position and orientation of new object
        :return: JSON of new scene state after object removal
        """
        object_spec = object_spec if isinstance(object_spec, EasyDict) else EasyDict(object_spec)
        placement_spec = placement_spec if isinstance(placement_spec, EasyDict) else EasyDict(placement_spec)
        updated_scene = self.object_placer.try_add(scene_state, object_spec, placement_spec)

        if object_spec.type == "category" and not placement_spec.get("allow_collisions"):
            tries = 1
            max_tries = 10
            while len(updated_scene.collisions) > 0 and tries < max_tries:
                print(f"has collisions {len(updated_scene.collisions)}, try different object {tries}/{max_tries}")
                updated_scene = self.object_placer.try_add(scene_state, object_spec, placement_spec)
                tries += 1
            print(f"placed after trying {tries} models to avoid collisions")
        else:  # no collision checking or specific object instance
            print(f"placed without collision checking")
        return updated_scene.to_json()

    def object_add_multiple(
        self, scene_state: JSONDict, specs: list[dict[str, ObjectSpec | PlacementSpec]]
    ) -> JSONDict:
        """Add new objects to existing scene (in bulk).

        :param scene_state: state of existing scene
        :param specs: specification describing objects to be added, as a list of dictionaries of the following form:
            {
                "object_spec": <ObjectSpec>,
                "placement_spec": <PlacementSpec>,
            }
        :return: JSON of new scene state after object removal
        """
        new_scene_state = scene_state
        for spec in specs:
            new_scene_state = self.object_add(new_scene_state, spec["object_spec"], spec["placement_spec"])
        return new_scene_state
