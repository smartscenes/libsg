"""
arch_builder.py
---
This code is intended to support generation and retrieval of the scene architecture.

Currently the code just performs a simple architecture retrieval in scene_builder.py, but this code will be fleshed
out in the near future.
"""

from libsg.assets import AssetDb
from libsg.scene_types import Arch


class ArchBuilder:
    def __init__(self):
        self.__arch_db: AssetDb  # pointer to arch db

    def generate(self, description: str) -> Arch:
        pass

    def modify(self, arch: Arch, description: str) -> Arch:
        pass

    def retrieve(self, description: str) -> Arch:
        pass
