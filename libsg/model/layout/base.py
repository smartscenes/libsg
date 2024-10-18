from typing import Any

from libsg.scene_types import SceneLayoutSpec


class BaseLayout:
    """
    Implements wrapper for scene layout generators.
    """

    def generate(self, layout_spec: SceneLayoutSpec, **kwargs) -> list[dict[str, Any]]:
        """
        Generates scene layout based on specification.

        :param layout_spec: specification for layout architecture
        :return: dictionary of the form
            {
                'class_labels': tensor of scores for classes for each object,
                'dimensions': tensor of dimensions for each object,
                'position': tensor of positions for each object,
                'orientation': tensor of angles to apply to each object,
            }
        """
        raise NotImplementedError
