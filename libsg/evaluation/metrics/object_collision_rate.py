import os

import pysolr
import trimesh

from .base import EvaluationBase
from libsg.scene import Scene


class ObjectLevelCollisionRate(EvaluationBase):
    def __init__(self, object_dir_mapping: str, solr_url: str):
        super().__init__()
        self.object_dir_mapping = object_dir_mapping
        self._solr = pysolr.Solr(solr_url) if solr_url else None

        self.num_collisions = 0
        self.num_objects = 0

    def _load_scene(self, scene: Scene):
        # query = f'fullId:({" OR ".join(scene.get_all_model_ids())})'
        # results = self._solr.search(query, rows=25000)

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

    def __call__(self, inp, scene_graph, scene: Scene):
        collision_manager = self._load_scene(scene)
        _, contact_pairs = collision_manager.in_collision_internal(return_names=True, return_data=False)

        objects_in_collision = set()
        for obj_a, obj_b in contact_pairs:
            objects_in_collision.add(obj_a)
            objects_in_collision.add(obj_b)

        self.num_collisions += len(objects_in_collision)
        self.num_objects += len(scene.model_instances)

    def log(self):
        result = self.num_collisions / self.num_objects
        print(f"Object-level collision rate: {result}")
        return result
