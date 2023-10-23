from libsg.assets import AssetDb
from libsg.scene_types import ObjectSpec, SceneContext


class ObjectSuggestor:
    def __init__(self):
        self.__model_db: AssetDb
    pass

    def suggest(self, object_spec: ObjectSpec, scene_context: SceneContext = None):
        pass
