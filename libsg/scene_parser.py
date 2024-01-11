from libsg.scene_types import SceneSpec, SceneType


class SceneParser:
    """
    Parse an unstructured scene description into a structured scene specification which can be used by downstream
    modules.

    The current implementation of this class involves very basic room parsing for the layout module, but in the future
    we would want to have more sophisticated parsing of the scene type, shape, and objects generated, such as with a
    scene graph or other representation.
    """

    def parse(self, scene_spec: SceneSpec) -> SceneSpec:
        """Parse scene description into a structured scene specification.

        The parsing function currently expects a "text" type scene specification and expects to find one of the
        following key phrases—"living room", "dining room", or "bedroom"—in the text description, which is used to
        specify the scene room type to generate. As of now, the scene parsing is fairly rudimentary and does not use
        any other information in the scene specification.

        :param scene_spec: unstructured scene specification
        :raises ValueError: scene spec type or room type not supported for parsing
        :return: structured scene specification
        """
        if scene_spec.type == SceneType.text:
            if "living room" in scene_spec.input:
                return SceneSpec(type=SceneType.category, input="living_room", format=scene_spec.format)
            elif "dining room" in scene_spec.input:
                return SceneSpec(type=SceneType.category, input="dining_room", format=scene_spec.format)
            elif "bedroom" in scene_spec.input:
                return SceneSpec(type=SceneType.category, input="bedroom", format=scene_spec.format)
            else:
                raise ValueError(f"Cannot parse room type from scene specification: {scene_spec.input}")
        else:
            raise ValueError(f"Cannot parse scene type from scene specification: {scene_spec.type}")
