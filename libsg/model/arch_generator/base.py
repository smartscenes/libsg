from libsg.scene_types import ArchSpec
from libsg.arch import Architecture


class BaseArchGenerator:
    """
    Parse an unstructured scene description into a structured scene specification which can be used by downstream
    modules.
    """

    def generate(self, scene_spec: ArchSpec) -> Architecture:
        """Generate an architecture based on the given specification.

        :param scene_spec: unstructured scene specification
        :raises ValueError: scene spec type not supported for generating architecture
        :return:
        """
        raise NotImplementedError
