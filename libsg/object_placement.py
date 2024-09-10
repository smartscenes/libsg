"""
object_placement.py
---
Code related to placement of objects in scenes.
"""

import copy
import logging
import random
import sys
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from libsg.assets import AssetDb
from libsg.geo import Transform
from libsg.scene_types import BBox3D, JSONDict, Point3D, ObjectSpec, PlacementSpec, SceneState
from libsg.scene import ModelInstance, Scene
from libsg.simulator import Simulator
from libsg.simscene import SimScene


@dataclass
class Placement:
    """Dataclass for object and its placement in scene"""

    position: Point3D = None
    up: Point3D = None
    front: Point3D = None
    object: ModelInstance = None
    ref_object: Union[ModelInstance, str] = None
    spec: PlacementSpec = None


class ObjectPlacer:
    """Place objects in scene with collision avoidance."""

    def __init__(self, model_db: AssetDb, size_threshold=0.5, allow_collisions: bool = True, max_tries: int = 1):
        self.__model_db = model_db
        self.size_threshold = size_threshold
        self.allow_collisions = allow_collisions
        self.max_tries = max_tries

    def _resolve_placement_spec(
        self, placement_spec: PlacementSpec, scene: Scene, object_to_place: ModelInstance
    ) -> Placement:
        placement = Placement(object=object_to_place, spec=placement_spec)
        placement.position = placement_spec.position
        placement.front = Transform.get_rotated_vector([0, 0, placement_spec.orientation], [1, 0, 0])

        if placement_spec.type == "placement_relation":
            placement.ref_object = self.resolve_object_spec_to_element_in_scene(scene, placement_spec.reference)
            if placement_spec.relation == "next":
                placement = self._place_next(scene, placement)
            elif placement_spec.relation == "on":
                placement = self._place_on(scene, placement)
        # else:
        #     print(f"Unsupported placement_spec.type={placement_spec.type}", file=sys.stderr)
        return placement

    def _place_on(self, scene: Scene, placement: Placement) -> Placement:
        sim = Simulator(mode="direct", verbose=False, use_y_up=False)
        sim_scene = SimScene(sim, scene, self.__model_db.config)
        if placement.ref_object.type == "ModelInstance":
            print(
                f"Place {placement.object.model_id} {placement.spec.relation} {placement.ref_object.model_id}",
                file=sys.stderr,
            )
            placement.position = self._sample_position_on(sim_scene, placement.ref_object.id)
        else:
            bbox3D = placement.ref_object.bbox3D
            print(
                f"Place {placement.object.model_id} {placement.spec.relation} {placement.ref_object.type} with {bbox3D}",
                file=sys.stderr,
            )
            obj_metadata = self.__model_db.get_metadata(placement.object.model_id)
            m = [obj_metadata.dims[0], obj_metadata.dims[1]]
            if placement.ref_object.type == "Wall":
                placement.front = placement.ref_object.front.tolist()
                placement.up = placement.ref_object.up.tolist()
                placement.position = placement.ref_object.sample_face(margin=m).tolist()
            elif placement.ref_object.type == "Floor":
                if not placement.position:  # sample a position
                    placement.position = bbox3D.sample_face(index=BBox3D.TOP, margin=m).tolist()
                else:  # drop from pre-specified position
                    placement.position = self._drop_place(sim_scene, placement.position)
                if not placement.front:
                    t = 2 * np.pi * np.random.random()
                    placement.front = Transform.get_rotated_vector([0, 0, t], [1, 0, 0])
            elif placement.ref_object.type == "Ceiling":
                placement.position = bbox3D.sample_face(index=BBox3D.BOTTOM, margin=m).tolist()
            else:
                print(f"Unsupported ref_object.type={placement.ref_object.type}", file=sys.stderr)
        return placement

    def _place_next(self, scene: Scene, placement: Placement) -> Placement:
        # get bbox of ref object
        with Simulator(mode="direct", verbose=False, use_y_up=False) as sim:
            sim_scene = SimScene(sim, scene, self.__model_db.config, include_ground=True)

            ref_obj_bbox = sim.get_aabb(placement.ref_object.id)
            obj_bbox = sim.get_aabb(placement.object.id)
            scene_bbox = sim.get_aabb_all()

            placement_found = False
            tries = 1
            max_tries = 10
            while not placement_found and tries < max_tries:
                # sample position on one of side faces or front face of bbox (TODO use semantic front)
                side = random.choice([BBox3D.LEFT, BBox3D.RIGHT, BBox3D.FRONT, BBox3D.BACK])
                print(f"sample side {side} {ref_obj_bbox}")
                sidepoint = ref_obj_bbox.sample_face(side, margin=[0, 0])

                # go out along face normal by [1, 2.5] * half-width
                obj_side_h = obj_bbox.get_face_dims(side)[2] * 0.5
                side_outnormal = ref_obj_bbox.get_face_outnormal(side)
                offset = side_outnormal.scale(random.uniform(1, 2.5) * obj_side_h)
                position = Point3D.add(sidepoint, offset)
                if not scene_bbox.contains(position):
                    print("sampled sidepoint outside scene, retrying...")
                    tries += 1
                    continue

                # find parent supporting ref_object with raycast (TODO use support hierarchy instead)
                ref_bottom = ref_obj_bbox.sample_face(BBox3D.BOTTOM, margin=[0, 0]).tolist()
                ref_bottom[2] -= 0.0001
                below_ref_bottom = list(ref_bottom)
                below_ref_bottom[2] -= 0.2
                parent_intersection = sim.ray_test(ref_bottom, below_ref_bottom)
                print("ref_bottom intersection", ref_bottom, below_ref_bottom, parent_intersection)

                if parent_intersection.id == -1:  # no parent found, naive 0 height placement
                    position.z = 0.0
                    placement.position = position.tolist()
                    print("placement without parent", placement)
                else:  # parent was found, drop down to parent
                    placement.position = self._drop_place(sim_scene, position)
                    print("placement with parent", placement)

                placement_found = True

        return placement

    # Note: will return either object or architectural element
    def resolve_object_spec_to_element_in_scene(self, scene: Scene, object_spec: ObjectSpec):
        object = None
        if object_spec.type == "object_id":
            object = scene.get_element_by_id(object_spec.object)
        elif object_spec.type == "model_id":
            objects = scene.find_objects_by_model_ids([object_spec.object])
            object = next(iter(objects), None)
        elif object_spec.type == "category":
            if ObjectSpec.is_arch(object_spec):
                element_type = object_spec.object.capitalize()
                elements = scene.arch.find_elements_by_type(element_type)
                object = next(iter(elements), None)
            else:
                query = self.__model_db.get_query_for_ids(scene.get_all_model_ids())
                query = query + f" AND {object_spec.object}"
                fq = self.object_query_constraints(object_spec)
                results = self.__model_db.search(query, fl="fullId", fq=fq, rows=25000)
                model_ids = list([result["fullId"] for result in results])
                objects = scene.find_objects_by_model_ids(model_ids)
                object = next(iter(objects), None)
        else:
            print(f"Unsupported object_spec.type={object_spec.type}", file=sys.stderr)
        return object

    # find position that is statically supported by reference_object
    def _sample_position_on(self, simscene: SimScene, reference_object_id: str):
        ref_obj_bbox = simscene.sim.get_aabb(reference_object_id)
        ref_obj_height = ref_obj_bbox.dims[2]
        ref_obj_margin = [0.1 * ref_obj_bbox.dims[0], 0.1 * ref_obj_bbox.dims[1]]
        position = ref_obj_bbox.get_face_center(BBox3D.TOP).tolist()

        placement_found = False
        tries = 1
        max_tries = 10
        while not placement_found and tries < max_tries:
            sampled_point = ref_obj_bbox.sample_face(BBox3D.TOP, margin=ref_obj_margin)
            top_point = sampled_point.tolist()
            top_point[2] += 0.1
            bottom_point = sampled_point.tolist()
            bottom_point[2] -= ref_obj_height
            intersection = simscene.sim.ray_test(top_point, bottom_point)
            print(f"try drop place {tries}/{max_tries}: {sampled_point} {intersection}")
            if intersection.id == reference_object_id:
                position = intersection.position
                placement_found = True
            tries += 1

        return position

    def _drop_place(self, simscene: SimScene, start_position: Point3D):
        top_point = start_position.tolist()
        bottom_point = start_position.tolist()
        bottom_point[2] -= 3.0  # TODO more intelligent than just "3m down"
        intersection = simscene.sim.ray_test(top_point, bottom_point)
        position = intersection.position if (intersection.id != -1) else start_position
        return position

    def try_add(
        self,
        scene: Scene,
        object_instance: ModelInstance,
        placement_spec: PlacementSpec,
        attempts: Optional[int] = None,
    ) -> Scene:
        """Try to add object to scene. If placement_spec.allow_collisions is True, checks if object placement conflicts
        with existing objects and will retry up to max_tries if a collision is detected.

        :param scene_state: JSON description of scene
        :param object_spec: specification of object to add
        :param placement_spec: specification of position/orientation of object
        :param max_tries: maximum number of tries to place object
        :return: scene after adding object
        """

        if attempts is None:
            attempts = self.max_tries

        # Create copy of object and create proposed placement. Note that the placement is only "random" if the object
        # is being placed relative to another, else there is no reason to try more than once.
        object_proposal = copy.deepcopy(object_instance)
        object_proposal.id = scene.get_next_id()
        object_proposal.parent_id = object_instance.id

        placement = self._resolve_placement_spec(placement_spec, scene, object_proposal)
        object_proposal.transform.set_translation(placement.position)
        rotation = Transform.get_alignment_quaternion(
            object_proposal.up, object_proposal.front, placement.up or scene.up, placement.front or scene.front
        )
        object_proposal.transform.set_rotation(rotation)

        if not self.allow_collisions:
            ignore_object_ids = []
            if placement.ref_object is not None:
                ignore_object_ids.append(placement.ref_object.id)
            contacts = self.check_object_contacts(scene, object_proposal.id, ignore_object_ids)
            if len(contacts) > 0:
                if attempts > 1:
                    return self.place(scene, object_instance, placement_spec, attempts - 1)
                else:
                    logging.warning("Object {} could not be placed without collisions.", object_instance.id)
                    return None

        # add to scene
        scene.add(object_proposal, clone=False)
    
        # TODO: do we need to add a modification to the scene?
        # scene.modifications.extend([{"type": "added", "object": a.to_json()} for a in added])
        return object_proposal

    def check_object_contacts(
        self, scene: Scene, object_id: str, ignore_object_ids: Optional[list[str]] = None
    ) -> dict[tuple[str, str], Any]:
        """Checks collisions with object in scene.

        :param scene: scene state
        :param object_id: ID of object against which to check collisions
        :param ignore_object_ids: list of object IDs to ignore in collision check, defaults to None
        :return: dictionary mapping pairs of IDs with Contact object
        """
        with Simulator(mode="direct", verbose=False, use_y_up=False) as sim:
            sim_scene = SimScene(sim, scene, self.__model_db.config)
            sim.step()
            contacts = sim.get_contacts(obj_id_a=object_id)  # , include_collision_with_static=True)
        # contacts = sim.get_contacts(object_id, include_collision_with_static=True)
        # print(f'contacts: {contacts}')
        if ignore_object_ids is not None:
            contacts = {pair: contact for pair, contact in contacts.items() if pair[1] not in ignore_object_ids}
        # print(f'filtered contacts: {contacts} {ignore_object_ids}')
        return contacts
