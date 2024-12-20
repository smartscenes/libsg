"""
scene_builder.py
---
Main code for generating a scene from a given scene description. This code currently handles the entire overall
pipeline, including architecture generation/retrieval, layout generation, and object generation/retrieval. In the 
future, some of the functionality here may be split off into other classes.
"""

import json
import os
import compress_json
import numpy as np
import sys
from easydict import EasyDict
from omegaconf import DictConfig
from typing import Any, Optional

import numpy as np
import pandas as pd
import trimesh

from libsg.arch_builder import ArchBuilder
from libsg.arch import Architecture
from libsg.assets import AssetDb
from libsg.object_builder import ObjectBuilder
from libsg.object_placement import ObjectPlacer
from libsg.scene_types import (
    ArchSpec,
    BBox3D,
    JSONDict,
    SceneState,
    ObjectSpec,
    ObjectTemplateInstance,
    Point3D,
    SceneSpec,
    SceneLayoutSpec,
    SceneLayout,
    PlacementSpec,
)
from libsg.scene import Scene
from libsg.simscene import SimScene
from libsg.simulator import Simulator
from libsg.geo import Transform
from libsg.io import SceneExporter
from libsg.model import build_layout_model

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

    def __init__(self, cfg: DictConfig, arch: DictConfig, layout: DictConfig, **kwargs):
        """
        :param cfg: loaded config at conf/config.yaml:scene_builder
        :param layout: loaded config at conf/config.yaml:scene_builder.layout
        """
        self.layout_cfg = layout
        self.__base_solr_url = cfg.get("solr_url")
        self.__arch_builder = ArchBuilder(cfg, arch, **kwargs)
        self.__scene_db = AssetDb("scene", cfg.get("scene_db"))
        self.__model_db = AssetDb("model", cfg.get("model_db"), solr_url=f"{self.__base_solr_url}/models3d")
        self.scene_exporter = SceneExporter()
        self.__object_builder = ObjectBuilder(self.__model_db, cfg.get("model_db"), **kwargs)
        self.object_placer = ObjectPlacer(
            model_db=self.__model_db, size_threshold=cfg.model_db.get("size_threshold", 0.5)
        )

        self.infer_missing_objects = kwargs.get("sceneInference.inferMissingObjects", False)
        self.layout_model = kwargs.get("sceneInference.layoutModel", self.layout_cfg.default_model)
        pass_text = kwargs.get("sceneInference.passTextToLayout", "False").lower()
        if pass_text not in {"true", "false"}:
            raise ValueError(f"Invalid value for sceneInference.passTextToLayout: {pass_text}")
        self.pass_text = pass_text == "true" and self.layout_cfg.config[self.layout_model].can_condition_on_text
        # "generate", "retrieve"
        self.object_method = kwargs.get("sceneInference.object.genMethod", cfg.model_db.default_object_gen_method)
        # "id", "category", "embedding", "text"
        self.object_retrieve_type = kwargs.get(
            "sceneInference.object.retrieveType", cfg.model_db.default_object_retrieve_type
        )
        self.move_objects_to_floor = (
            kwargs.get(
                "sceneInference.layout.moveObjectsToFloor", "False" if self.layout_model == "Holodeck" else "True"
            ).lower()
            == "true"
        )

        # architecture parameters
        self.arch_method = kwargs.get("sceneInference.arch.genMethod", "retrieve")

    def modify(self, scene_state: JSONDict, description: str) -> JSONDict:
        raise NotImplementedError

    def generate(self, scene_spec: SceneSpec) -> JSONDict:
        """
        Main function for generating a complete 3D scene.

        This method orchestrates the entire scene generation process, including:
        1. Generating or retrieving the room architecture
        2. Creating a layout of objects within the room
        3. Generating or retrieving 3D meshes based on the layout
        4. Output the final scene

        :param scene_spec: A SceneSpec object containing the specification for the scene to generate.

        :return: A JSON dictionary representing the complete generated scene according to the specified output format (e.g., HAB or STK)

        :raises ValueError: If the scene specification is invalid or unsupported.
        """
        # generate or retrieve architecture
        arch_spec = (
            scene_spec.arch_spec
            if scene_spec.arch_spec is not None
            else ArchSpec(type="id", input=None, format=scene_spec.format, prompt=scene_spec.raw)
        )
        arch, scene, base_scene_aabb = self.generate_arch(arch_spec)
        base_scene_centroid = base_scene_aabb.centroid

        # generate layout
        scene_layout_spec = SceneLayoutSpec(
            type=scene_spec.type, input=scene_spec.input, arch=arch, graph=scene_spec.scene_graph
        )
        if self.pass_text:
            scene_layout_spec.raw = scene_spec.raw
        scene_layout = self.generate_layout(scene_layout_spec)

        # generate or retrieve objects
        scene = self.generate_objects(scene, base_scene_centroid, scene_layout)

        # format scene
        return self.export_scene(scene, scene_spec.format)

    def generate_arch(self, arch_spec: ArchSpec) -> tuple[Architecture, Scene, BBox3D]:
        """Generate architecture for scene.

        The current implementation retrieves an architecture from Structured3D.

        :param scene_spec: specification of scene. As of current, assumes that scene spec specifies a pre-existing
        structured3d ID to load as the base scene, without pre-existing objects.
        :return: tuple of architecture object, scene JSON (without objects), and axis-aligned bounding box around
        scene, as calculated by the min/max AABB of all elements in the scene.
        """
        # load base scene and clear default layout objects
        arch = getattr(self.__arch_builder, self.arch_method)(arch_spec, min_size=self.layout_cfg.min_room_size)
        base_scene = Scene.from_arch(arch)

        # compute base scene AABB to transform object positions later
        with Simulator(mode="direct", verbose=False, use_y_up=False) as sim:
            sim_scene = SimScene(sim, base_scene, self.__model_db.config, include_ground=True)
            base_scene_aabb = sim_scene.aabb

        return arch, base_scene, base_scene_aabb

    def generate_layout(self, layout_spec: SceneLayoutSpec) -> SceneLayout:
        """Generate coarse scene layout given specification of architecture and room type.

        :param layout_spec: specification for architecture and room type
        :param model_name: name of model. If not provided, the default model is pulled from the configuration.
        :return: scene layout, including architecture and coarse object types and poses
        """
        layout_model = build_layout_model(
            layout_spec, self.layout_model, self.layout_cfg, bounds=None, text_condition=self.pass_text
        )
        objects = layout_model.generate(layout_spec)

        layout = SceneLayout(
            objects=[
                ObjectTemplateInstance(
                    id=obj.get("id"),
                    label=obj["wnsynsetkey"],
                    dimensions=obj["dimensions"],
                    position=obj["position"],
                    orientation=obj["orientation"],
                    embedding=obj.get("embedding"),
                    description=obj.get("class"),
                )
                for obj in objects
            ],
            arch=layout_spec.arch,
            room_type=layout_spec.input,  # may become obsolete in the future
            shift_by_scene_centroid=layout_model.SHIFT_BY_SCENE_CENTROID,
        )

        # print layout
        layout.print_layout()

        if self.layout_cfg.export_layout:
            layout.export_coarse_layout("coarse_layout.obj")
        return layout

    def generate_objects(
        self,
        scene: Scene,
        base_scene_centroid: Point3D,
        scene_layout: SceneLayout,
    ) -> Scene:
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

            # TODO heuristically re-center to base scene centroid
            position = (
                Point3D.add(Point3D.fromlist(obj.position), base_scene_centroid)
                if scene_layout.shift_by_scene_centroid
                else Point3D.fromlist(obj.position)
            )
            if self.move_objects_to_floor:
                position.z = self.NEAR_FLOOR_HEIGHT  # TODO Heuristically shift to near-floor placement
                if synset is not None and ("pendant_lamp" in synset or "ceiling_lamp" in synset):
                    continue  # skip ceiling lamps  # TODO: ceiling lamp placement logic
            orientation = obj.orientation - np.pi / 2  # TODO figure out why theta adjustment seems to be required here
            object_spec = ObjectSpec(
                type=self.object_retrieve_type,
                wnsynsetkey=synset,
                dimensions=dimensions,
                embedding=obj.embedding,
                description=obj.id if self.object_retrieve_type == "id" else obj.description,
            )
            placement_spec = PlacementSpec(type="placement_point", position=position, orientation=orientation)

            # retrieve by embedding
            kwargs = {}
            if self.object_method == "retrieve" and object_spec.type == "embedding":
                kwargs["embedding_type"] = self.layout_cfg.config[self.layout_model].embedding_type
            kwargs["room_type"] = scene_layout.room_type

            # retrieve or generate object (either generate() or retrieve() on object_builder based on config)
            model_instance = getattr(self.__object_builder, self.object_method)(
                object_spec, placement_spec, dataset_name=scene_layout.room_type, **kwargs
            )

            # add object to scene
            if model_instance is not None:
                self.object_placer.try_add(scene, model_instance, placement_spec)

        return scene

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
            for m in scene.model_instances:
                m.transform.rightMultiply(preR)

            # export scene
            scene_json = self.scene_exporter.export(scene, format=SceneExporter.SceneFormat.HAB)
        elif format == "STK":
            scene_json = self.scene_exporter.export(scene, format=SceneExporter.SceneFormat.STK)
        else:
            raise ValueError(f"Scene format not supported: {format}")

        return scene_json

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

    # def object_remove(self, scene_state: SceneState, object_spec: ObjectSpec) -> JSONDict:
    #     """Remove existing object from given scene.

    #     :param scene_state: state of existing scene
    #     :param object_spec: specification describing object to be removed
    #     :return: JSON of new scene state after object removal
    #     """
    #     scene = Scene.from_json(scene_state)
    #     object_spec = object_spec if isinstance(object_spec, EasyDict) else EasyDict(object_spec)

    #     removed = []
    #     if object_spec.type == "object_id":
    #         id_to_remove = object_spec.object
    #         removed_obj = scene.remove_object_by_id(id_to_remove)
    #         if removed_obj is not None:
    #             removed.append(removed_obj)
    #     elif object_spec.type == "model_id":
    #         model_id_to_remove = object_spec.object
    #         removed = scene.remove_objects(lambda obj: obj.model_id == model_id_to_remove)
    #     elif object_spec.type == "category":
    #         fq = self.object_placer.object_query_constraints(object_spec)
    #         results = self.__model_db.search(object_spec.object, fl="fullId", fq=fq, rows=25000)
    #         model_ids = set([result["fullId"] for result in results])
    #         removed = scene.remove_objects(lambda obj: obj.model_id in model_ids)
    #     else:
    #         print(f"Unsupported object_spec.type={object_spec.type}", file=sys.stderr)

    #     scene.modifications.extend([{"type": "removed", "object": r.to_json()} for r in removed])
    #     return scene.to_json()
