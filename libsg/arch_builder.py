from libsg.assets import AssetDb
from libsg.scene_types import Arch


class ArchBuilder:
    def __init__(self):
        self.__arch_db: AssetDb   # pointer to arch db

    def generate(self, description: str) -> Arch:
        pass

    def modify(self, arch: Arch, description: str) -> Arch:
        pass

    def retrieve(self, description: str) -> Arch:
        pass
