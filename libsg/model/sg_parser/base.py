from libsg.scene_types import SceneSpec


class BaseSceneParser:
    """
    Parse an unstructured scene description into a structured scene specification which can be used by downstream
    modules.
    """

    def parse(self, scene_spec: SceneSpec) -> SceneSpec:
        """Parse scene description into a structured scene specification.

        :param scene_spec: unstructured scene specification
        :raises ValueError: scene spec type or room type not supported for parsing
        :return: structured scene specification
        """
        raise NotImplementedError
