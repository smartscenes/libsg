import os

import pysolr
import trimesh

from .base import EvaluationBase
from libsg.scene import Scene
from libsg.scene_types import Floor, BBox3D


class SceneLevelInBoundsRate(EvaluationBase):
    def __init__(self, object_dir_mapping: str, solr_url: str, wall_depth: float = 0.0):
        super().__init__()
        self.object_dir_mapping = object_dir_mapping
        self._solr = pysolr.Solr(solr_url) if solr_url else None
        self.wall_depth = wall_depth

        self.num_in_bounds = 0
        self.num_scenes = 0

    def _load_arch(self, scene: Scene) -> list[Floor]:
        arch = scene.arch
        return list(arch.find_elements_by_type("Floor"))
    
    def _in_bounds(self, floors: list[Floor], model_aabb: BBox3D) -> bool:
        for floor in floors:
            if floor.contains(model_aabb, wall_depth=self.wall_depth):
                return True
        return False

    def _load_scene(self, scene: Scene):
        # query = f'fullId:({" OR ".join(scene.get_all_model_ids())})'
        # results = self._solr.search(query, rows=25000)

        models = []
        for mi in scene.model_instances:
            # TODO: change this to a solr query?
            # TODO: make this more general to support other asset sources
            asset_source, _, model_id = mi.model_id.partition(".")
            object_path = os.path.join(
                self.object_dir_mapping[asset_source], str(model_id[0]), f"{model_id}.glb"
            )

            model_mesh = trimesh.load(object_path, force="mesh")
            bounds = model_mesh.bounds
            model_aabb = BBox3D.from_min_max(bounds[0], bounds[1])
            models.append(model_aabb)

        return models

    def __call__(self, inp, scene_graph, scene: Scene):
        self.num_scenes += 1
        floors = self._load_arch(scene)
        models = self._load_scene(scene)

        # TODO: add checking for vertical bounds
        for m in models:
            if not self._in_bounds(floors, m):
                return False
        
        self.num_in_bounds += 1
        return True


    def log(self):
        result = self.num_in_bounds / self.num_scenes
        print(f"Scene-level in-bounds rate: {result}")
        return result

