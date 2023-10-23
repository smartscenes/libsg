import copy
from libsg.arch import Architecture
from libsg.geo import Transform
from libsg.scene_types import JSONDict


class ModelInstance:
    def __init__(self, model_id='', transform=Transform(), id='', parent_id=None):
        self.id: str = id
        self.type: str = 'ModelInstance'
        self.model_id: str = model_id
        self.parent_id: str = parent_id
        self.transform: Transform = copy.deepcopy(transform)
        self.metadata: dict = None

    def to_json(self, obj=None) -> JSONDict:
        obj = obj if obj else {}
        obj['modelId'] = self.model_id
        obj['id'] = self.id
        if self.parent_id is not None:
            obj['parentId'] = self.parent_id
        obj['transform'] = self.transform.to_json()
        return obj

    @classmethod
    def from_json(cls, obj):
        mi = ModelInstance()
        mi.model_id = obj['modelId']
        if 'id' in obj:
            mi.id = obj['id']
        else:
            mi.id = str(obj['index'])
        mi.parent_id = obj.get('parentId')
        mi.transform = Transform.from_json(obj['transform'])
        return mi


class Scene:
    def __init__(self, id='', asset_source=None):
        self.id = id
        self.asset_source = asset_source
        self.version = 'scene@1.0.2'
        self.up = [0, 0, 1]
        self.front = [0, 1, 0]
        self.unit = 1
        self.arch = None
        self.modelInstances_by_id: dict[str,ModelInstance] = {}
        self.modifications = []
        self.collisions = []
        self.__maxId = 0

    @property
    def modelInstances(self):
        return self.modelInstances_by_id.values()

    def __getNextId(self):
        self.__maxId = self.__maxId + 1
        return f'{self.__maxId}'

    def add(self, mi: ModelInstance, clone=False, parent_id=None) -> ModelInstance:
        if clone:
            modelInst = copy.deepcopy(mi)
            modelInst.id = self.__getNextId()
            modelInst.parent_id = parent_id
        else:
            modelInst = mi
        if modelInst.id in self.modelInstances_by_id:
            raise Exception('Attempting to add model instance with duplicate id to scene') 
        self.modelInstances_by_id[modelInst.id] = modelInst
        if modelInst.id.isdigit():
            self.__maxId = max(int(modelInst.id), self.__maxId)
        return modelInst

    def get_object_by_id(self, id):
        return self.modelInstances_by_id.get(id)

    def get_arch_element_by_id(self, id):
        return self.arch.get_element_by_id(id)

    def get_element_by_id(self, id):
        element = self.get_object_by_id(id)
        if element is None:
            element = self.get_arch_element_by_id(id)
        return element

    def find_objects(self, cond):
        return filter(cond, self.modelInstances_by_id.values())

    def find_objects_by_model_ids(self, model_ids):
        return filter(lambda x: x.model_id in model_ids, self.modelInstances_by_id.values())

    def get_all_model_ids(self):
        return list(set([m.model_id for m in self.modelInstances_by_id.values()]))

    def remove_object_by_id(self, id):
        removed = None
        if id in self.modelInstances_by_id:
            removed = self.modelInstances_by_id[id]
            del self.modelInstances_by_id[id]
        return removed

    def remove_objects(self, cond):
        removed = list(filter(cond, self.modelInstances_by_id.values()))
        for r in removed:
            self.remove_object_by_id(r.id)
        return removed

    def set_arch(self, a):
        self.arch = a

    def to_json(self, obj=None) -> JSONDict:
        obj = obj if obj else {}
        obj['version'] = self.version
        obj['id'] = self.id
        obj['up'] = self.up
        obj['front'] = self.front
        obj['unit'] = self.unit
        obj['assetSource'] = self.asset_source
        obj['arch'] = self.arch.to_json()
        obj['object'] = [mi.to_json() for mi in self.modelInstances]
        obj['modifications'] = self.modifications

        # add index (perhaps not needed in future)
        id_to_index = { mi['id'] : i for i,mi in enumerate(obj['object'])}
        for i,mi in enumerate(obj['object']):
            mi['index'] = i
            if mi.get('parentId') is not None:
                mi['parentIndex'] = id_to_index[mi['id']] 
        return obj

    @classmethod
    def from_json(cls, obj):
        scn = Scene()
        scn.id = obj['id']
        scn.asset_source = obj['assetSource']
        scn.up = list(obj['up'])
        scn.front = list(obj['front'])
        scn.unit = float(obj['unit'])
        scn.arch = Architecture.from_json(obj['arch'])
        scn.arch.ensure_typed()
        scn.modifications = obj.get('modifications', [])
        scn.modelInstances_by_id = {}
        for o in obj['object']:
            scn.add(ModelInstance.from_json(o)) 
        return scn
