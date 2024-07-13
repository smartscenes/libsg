import copy
from typing import Self

import numpy as np
from scipy.spatial.transform import Rotation

from libsg.arch import Architecture
from libsg.geo import Transform
from libsg.scene_types import JSONDict


class ModelInstance:
    """Main definition of a model/object and its associated transform, related objects, and metadata"""

    def __init__(self, model_id="", transform=Transform(), id="", parent_id=None, up=None, front=None):
        self.id: str = id
        self.type: str = "ModelInstance"
        self.model_id: str = model_id
        self.parent_id: str = parent_id
        self.transform: Transform = copy.deepcopy(transform)
        self.metadata: dict = None
        self.up = up
        self.front = front

    def to_json(self, obj=None) -> JSONDict:
        obj = obj if obj else {}
        obj["modelId"] = self.model_id
        obj["id"] = self.id
        if self.parent_id is not None:
            obj["parentId"] = self.parent_id
        obj["transform"] = self.transform.to_json()
        return obj

    @classmethod
    def from_json(cls, obj) -> Self:
        mi = ModelInstance()
        mi.model_id = obj["modelId"]
        if "id" in obj:
            mi.id = obj["id"]
        else:
            mi.id = str(obj["index"])
        mi.parent_id = obj.get("parentId")
        mi.transform = Transform.from_json(obj["transform"])
        return mi

    def swap_axes(
        self,
        orig_up: list[int],
        orig_front: list[int],
        target_up: list[int],
        target_front: list[int],
        invert: bool,
        rotate: float,
    ) -> Self:
        """Transform coordinate system of object in scene.

        :param orig_up: original upward-facing axis, in the form of a one-hot vector. This method does NOT support
        arbitrary axis directions.
        :param orig_front: original front-facing axis, in the form of a one-hot vector. This method does NOT support
        arbitrary axis directions.
        :param target_up: new upward-facing axis, in the form of a one-hot vector. This method does NOT support
        arbitrary axis directions.
        :param target_front: new front-facing axis, in the form of a one-hot vector. This method does NOT support
        arbitrary axis directions.
        :param invert: if True, invert the target_front axis after swapping in order to switch from LHS to RHS (or vice
        versa)
        :param rotate: if True, rotate the object around the target_up axis by the specified angle
        :return: self
        """
        self.transform.swap_axes(orig_up, target_up)

        # if the original front axis is now the up axis, update the original front to the axis which took its place
        if orig_front == target_up:
            orig_front == orig_up
        self.transform.swap_axes(orig_front, target_front)

        if invert:  # convert back and forth between LH and RH axis
            self.transform.invert_axis(target_front)

        if rotate != 0:
            self.transform.rotate(target_up, rotate)

        return self


class Scene:
    """Main scene definition"""

    def __init__(self, id="", asset_source=None):
        self.id = id
        self.asset_source = asset_source
        self.version = "scene@1.0.2"
        self.up = [0, 0, 1]
        self.front = [0, 1, 0]
        self.unit = 1
        self.arch = None
        self.model_instances_by_id: dict[str, ModelInstance] = {}
        self.modifications = []
        self.__maxId = 0

    @property
    def model_instances(self):
        return self.model_instances_by_id.values()

    def get_next_id(self):
        self.__maxId = self.__maxId + 1
        return f"{self.__maxId}"

    def add(self, mi: ModelInstance, clone=False, parent_id=None) -> ModelInstance:
        if clone:
            modelInst = copy.deepcopy(mi)
            modelInst.id = self.__getNextId()
            modelInst.parent_id = parent_id
        else:
            modelInst = mi
        if modelInst.id in self.model_instances_by_id:
            raise Exception("Attempting to add model instance with duplicate id to scene")
        self.model_instances_by_id[modelInst.id] = modelInst
        if modelInst.id.isdigit():
            self.__maxId = max(int(modelInst.id), self.__maxId)
        return modelInst

    def get_object_by_id(self, id):
        return self.model_instances_by_id.get(id)

    def get_arch_element_by_id(self, id):
        return self.arch.get_element_by_id(id)

    def get_element_by_id(self, id):
        element = self.get_object_by_id(id)
        if element is None:
            element = self.get_arch_element_by_id(id)
        return element

    def find_objects(self, cond):
        return filter(cond, self.model_instances_by_id.values())

    def find_objects_by_model_ids(self, model_ids):
        return filter(lambda x: x.model_id in model_ids, self.model_instances_by_id.values())

    def get_all_model_ids(self):
        return list(set([m.model_id for m in self.model_instances_by_id.values()]))

    def remove_object_by_id(self, id):
        removed = None
        if id in self.model_instances_by_id:
            removed = self.model_instances_by_id[id]
            del self.model_instances_by_id[id]
        return removed

    def remove_objects(self, cond):
        removed = list(filter(cond, self.model_instances_by_id.values()))
        for r in removed:
            self.remove_object_by_id(r.id)
        return removed

    def set_arch(self, a):
        self.arch = a

    def set_axes(self, up: list[int], front: list[int], invert: bool = False, rotate: float = 0) -> Self:
        """Transform coordinate system of object in scene.

        :param up: new upward-facing axis, in the form of a one-hot vector. This method does NOT support
        arbitrary axis directions.
        :param front: new front-facing axis, in the form of a one-hot vector. This method does NOT support
        arbitrary axis directions.
        :param invert: if True, invert the target_front axis after swapping in order to switch from LHS to RHS (or vice
        versa)
        :param rotate: if True, rotate the object around the target_up axis by the specified angle
        :return: self
        """

        def get_axis_index(axis_vec):
            return np.array([0, 1, 2])[list(map(bool, axis_vec))].item()

        if self.front == front and self.up == up:
            return self
        else:
            orig_up = get_axis_index(self.up)
            orig_front = get_axis_index(self.front)
            up_axis = get_axis_index(up)
            front_axis = get_axis_index(front)

            # update arch
            self.arch.set_axes(up_axis, front_axis, invert=invert, rotate=rotate)

            # update objects
            for obj in self.model_instances:
                obj.swap_axes(orig_up, orig_front, up_axis, front_axis, invert=invert, rotate=rotate)

            # update modifications
            if len(self.modifications) > 0:
                raise NotImplementedError("The set_axes operation does not yet support modifications")

            # update directions
            self.up = up
            self.front = front

        return self

    def to_json(self, obj=None) -> JSONDict:
        obj = obj if obj else {}
        obj["version"] = self.version
        obj["id"] = self.id
        obj["up"] = self.up
        obj["front"] = self.front
        obj["unit"] = self.unit
        obj["assetSource"] = self.asset_source
        obj["arch"] = self.arch.to_json()
        obj["object"] = [mi.to_json() for mi in self.model_instances]
        obj["modifications"] = self.modifications

        # add index (perhaps not needed in future)
        id_to_index = {mi["id"]: i for i, mi in enumerate(obj["object"])}
        for i, mi in enumerate(obj["object"]):
            mi["index"] = i
            if mi.get("parentId") is not None:
                mi["parentIndex"] = id_to_index[mi["id"]]
        return obj

    @classmethod
    def from_json(cls, obj) -> Self:
        scn = Scene()
        scn.id = obj["id"]
        scn.asset_source = obj["assetSource"]
        scn.up = list(obj["up"])
        scn.front = list(obj["front"])
        scn.unit = float(obj["unit"])
        scn.arch = Architecture.from_json(obj["arch"])
        scn.arch.ensure_typed()
        scn.modifications = obj.get("modifications", [])
        scn.model_instances_by_id = {}
        for o in obj["object"]:
            scn.add(ModelInstance.from_json(o))
        return scn

    @classmethod
    def from_arch(cls, arch: Architecture, asset_source: str) -> Self:
        """Generate a scene based on an architecture (i.e. without objects)

        :param arch: architecture to use in scene
        :param asset_source: source of assets
        :return: new Scene object
        """
        scn = Scene()
        scn.id = str(arch.id)
        scn.asset_source = asset_source
        scn.up = arch.up
        scn.front = arch.front
        scn.unit = float(arch.unit)
        scn.arch = arch
        scn.modifications = []
        scn.model_instances_by_id = {}
        return scn
