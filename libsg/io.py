import numpy as np
from enum import Enum
from libsg.scene import Scene
from libsg.scene_types import JSONDict


class SceneExporter:
    """
    Export scene to specified scene format. Current supported formats include HAB (habitat) and STK (scene toolkit)
    """

    class SceneFormat(Enum):
        STK = 1
        HAB = 2

    def export(self, scene: Scene, format: SceneFormat, obj=None) -> JSONDict:
        """Export scene to scene format

        :param scene: scene to export
        :param format: format to which to export scene (HAB or STK)
        :param obj: existing JSON to which to add scene format, defaults to None (start JSON from scratch)
        :return: _description_
        """
        obj = obj if obj else {}
        if format == SceneExporter.SceneFormat.STK:
            return self._export_stk(scene, obj)
        elif format == SceneExporter.SceneFormat.HAB:
            return self._export_hab(scene, obj)
        else:
            print(f"Unknown format {format}")
            exit(-1)

    def _export_stk(self, scene: Scene, obj: JSONDict) -> JSONDict:
        obj["format"] = "sceneState"
        obj["scene"] = scene.to_json()
        obj["selected"] = []
        return obj

    def _export_hab(self, scene: Scene, obj: JSONDict) -> JSONDict:
        obj["stage_instance"] = {"template_name": f"stages/{scene.id}"}
        obj["translation_origin"] = "asset_local"
        object_instances = []
        for mi in scene.model_instances:
            o_obj = {}
            raw_id = mi.model_id.split(".")[1]
            o_obj["template_name"] = raw_id
            o_obj["translation"] = list(mi.transform.translation)
            o_obj["rotation"] = list(np.roll(mi.transform.rotation, 1))  # xyzw -> wxyz
            o_obj["non_uniform_scale"] = list(mi.transform.scale)
            o_obj["motion_type"] = "STATIC"
            object_instances.append(o_obj)
        obj["object_instances"] = object_instances
        return obj
