import logging
import os

import trimesh

from .base import EvaluationBase
from libsg.scene import Scene


class SceneLevelCollisionRate(EvaluationBase):
    def __init__(self, object_dir_mapping: str):
        super().__init__()
        self.object_dir_mapping = object_dir_mapping

        self.num_collisions = 0
        self.num_scenes = 0

    def _load_scene(self, scene: Scene):
        collision_manager = trimesh.collision.CollisionManager()
        for mi in scene.model_instances:
            # TODO: change this to a solr query?
            # TODO: make this more general to support other asset sources
            asset_source, _, model_id = mi.model_id.partition(".")
            object_path = os.path.join(
                self.object_dir_mapping[asset_source], str(model_id[0]), f"{model_id}.glb"
            )

            model_mesh = trimesh.load(object_path, force="mesh")
            collision_manager.add_object(mi.model_id, model_mesh, transform=mi.transform.mat4)

        return collision_manager

    def __call__(self, inp, scene_graph, scene: Scene, **kwargs):
        collision_manager = self._load_scene(scene)
        is_collision = collision_manager.in_collision_internal(return_names=False, return_data=False)

        self.num_collisions += int(is_collision)
        self.num_scenes += 1

    def log(self):
        result = self.num_collisions / self.num_scenes
        print(f"Scene-level collision rate: {result}")
        return result
