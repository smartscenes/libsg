from libsg.assets import AssetDb
from libsg.object_placement import ObjectPlacer
from libsg.scene_types import SceneState, ObjectSpec, Point3D, SceneSpec, PlacementSpec
from libsg.scene import Scene
from libsg.simscene import SimScene
from libsg.simulator import Simulator
from libsg.geo import Transform
from libsg.io import SceneExporter
from libsg.model.atiss import atiss_network, descale_bbox_params, square_room_mask
from libsg.config import config as cfg

import json
import sys
from easydict import EasyDict
from typing import List, Tuple


class SceneBuilder:
    def __init__(self, cfg):
        self.__base_solr_url = cfg.get('solr_url')
        self.__arch_db = AssetDb(cfg.get('arch_db'))
        self.__scene_db = AssetDb(cfg.get('scene_db'))
        self.__model_db = AssetDb(cfg.get('model_db'), solr_url = f'{self.__base_solr_url}/models3d')
        self.scene_exporter = SceneExporter()
        self.object_placer = ObjectPlacer(model_db = self.__model_db)

    def generate(self, scene_spec: SceneSpec) -> SceneState:
        if scene_spec.type == 'id':
            # load base scene and clear default layout objects
            base_scene_path = self.__scene_db.get(scene_spec.input)
            base_scene = json.load(open(base_scene_path, 'r'))['scene']
            base_scene['object'] = []  # keep arch, drop existing objects

            # compute base scene AABB to transform object positions later
            with Simulator(mode='direct', verbose=False, use_y_up=False) as sim:
                sim_scene = SimScene(sim, Scene.from_json(base_scene), self.__model_db.config, include_ground=True)
                base_scene_aabb = sim_scene.aabb

            # initialize generator
            room_dim = int(cfg.atiss.data.room_layout_size.split(',')[0])
            classes = list(cfg.atiss.data.classes)
            bounds = cfg.atiss.data.bounds
            atiss = atiss_network(config=cfg.atiss)
            atiss.eval()

            # call generator and parse output objects
            room_mask = square_room_mask(room_dim=room_dim)
            bbox_params = atiss.generate_boxes(room_mask=room_mask)
            boxes = descale_bbox_params(bounds, bbox_params)
            objects = []
            for i in range(1, boxes['class_labels'].shape[1]-1):
                objects.append({
                    'wnsynsetkey': classes[boxes['class_labels'][0, i].argmax(-1)],
                    'dimensions': boxes['sizes'][0, i, :].tolist(),
                    'position': boxes['translations'][0, i, :].tolist(),
                    'orientation': float(boxes['angles'][0, i, 0]),
                })
            # print(objects)

            # create ObjectSpec and PlacementSpec for each object and place in scene
            scene_state = base_scene
            base_scene_centroid = base_scene_aabb.centroid
            for i in range(len(objects)):
                synset = objects[i]['wnsynsetkey']
                dimensions = objects[i]['dimensions']
                position = Point3D.add(Point3D.fromlist(objects[i]['position']), base_scene_centroid)  # TODO heuristically re-center to base scene centroid
                position.z = 0.05  # TODO Heuristically shift to near-floor placement
                orientation = objects[i]['orientation'] - 3.14159 / 2  # TODO figure out why theta adjustment seems to be required here
                if 'pendant_lamp' in synset or 'ceiling_lamp' in synset:
                    continue  # skip ceiling lamps  # TODO: ceiling lamp placement logic
                object_spec = {'type': 'category', 'wnsynsetkey': synset, 'dimensions': dimensions}
                placement_spec = {'type': 'placement_point', 'position': position, 'orientation': orientation}
                scene_state = self.object_add(scene_state, object_spec, placement_spec)

            scene = Scene.from_json(scene_state)
            if scene_spec.format == 'HAB':
                # Habitat object instance pre-rotation
                preRquat = Transform.get_alignment_quaternion([0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, -1])
                preR = Transform()
                preR.set_rotation(preRquat)
                for m in scene.modelInstances:
                    m.transform.rightMultiply(preR)

                # export scene
                scene_json = self.scene_exporter.export(scene, format=SceneExporter.SceneFormat.HAB)
            else:
                scene_json = self.scene_exporter.export(scene, format=SceneExporter.SceneFormat.STK)

            # return scene
            return scene_json

    def modify(self, scene_state: SceneState, description: str) -> SceneState:
        pass

    def retrieve(self, scene_spec: SceneSpec) -> SceneState:
        if scene_spec.type == 'id':
            scenestate_path = self.__scene_db.get(scene_spec.input)
            scenestate = json.load(open(scenestate_path, 'r'))
            return scenestate

    def object_remove(self, scene_state: SceneState,
                      object_spec: ObjectSpec) -> SceneState:
        scene = Scene.from_json(scene_state)
        object_spec = object_spec if isinstance(object_spec, EasyDict) else EasyDict(object_spec)

        removed = []
        if object_spec.type == 'object_id':
            id_to_remove = object_spec.object
            removed_obj = scene.remove_object_by_id(id_to_remove)
            if removed_obj is not None:
                removed.append(removed_obj)
        elif object_spec.type == 'model_id':
            model_id_to_remove = object_spec.object
            removed = scene.remove_objects(lambda obj: obj.model_id == model_id_to_remove)
        elif object_spec.type == 'category':
            fq = self.object_placer.object_query_constraints(object_spec)
            results = self.__model_db.search(object_spec.object, fl='fullId', fq=fq, rows=25000)
            model_ids = set([result['fullId'] for result in results])
            removed = scene.remove_objects(lambda obj: obj.model_id in model_ids)
        else:
            print(f'Unsupported object_spec.type={object_spec.type}', file=sys.stderr)

        scene.modifications.extend([{'type': 'removed', 'object': r.to_json()} for r in removed])
        return scene.to_json()

    def object_add(self, scene_state: SceneState,
                         object_spec: ObjectSpec,
                         placement_spec: PlacementSpec) -> SceneState:

        object_spec = object_spec if isinstance(object_spec, EasyDict) else EasyDict(object_spec)
        placement_spec = placement_spec if isinstance(placement_spec, EasyDict) else EasyDict(placement_spec)
        updated_scene = self.object_placer.try_add(scene_state, object_spec, placement_spec)

        if object_spec.type == 'category' and not placement_spec.get('allow_collisions'):
            tries = 1
            max_tries = 10
            while len(updated_scene.collisions) > 0 and tries < max_tries:
                print(f'has collisions {len(updated_scene.collisions)}, try different object {tries}/{max_tries}')
                updated_scene =  self.object_placer.try_add(scene_state, object_spec, placement_spec)
                tries += 1
            print(f'placed after trying {tries} models to avoid collisions')
        else:  # no collision checking or specific object instance
            print(f'placed without collision checking')
        return updated_scene.to_json()

    def object_add_multiple(self, scene_state: SceneState,
                            specs: List[Tuple[ObjectSpec,PlacementSpec]]) -> SceneState:
        new_scene_state = scene_state
        for spec in specs:
            new_scene_state = self.object_add(new_scene_state, spec['object_spec'], spec['placement_spec'])
        return new_scene_state
