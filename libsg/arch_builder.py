"""
arch_builder.py
---
This code is intended to support generation and retrieval of the scene architecture.
"""

import json
import random
from typing import Optional

import numpy as np
import requests
from omegaconf import DictConfig
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from libsg.assets import AssetDb
from libsg.arch import Architecture


class ArchBuilder:
    """
    Build or retrieve architecture for scenes.
    """

    DEFAULT_UP = [0, 0, 1]
    DEFAULT_FRONT = [0, 1, 0]
    LAST_ARCH = None

    def __init__(self, cfg: DictConfig):
        self.__arch_db = AssetDb("arch", cfg)

        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1)
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def generate(self, description: str) -> Architecture:
        pass

    def modify(self, arch: Architecture, description: str) -> Architecture:
        pass

    def retrieve(self, description: Optional[str] = None, whole_scene: bool = False, min_size: float = 0.0) -> Architecture:
        """Retrieve architecture from database.

        Because structured3d scenes have the y-axis as upward and use a left-handed coordinate system (LHS), the code
        ensures that the original architecture is transformed such that the z-axis is upward, the y-axis is the front
        axis, and the LHS is transformed to RHS (flips the y-axis and then rotates by pi radians about the z-axis in
        order to align it with the expected coordinate system orientation.)

        If multiple rooms are present in the scene, a random room is selected.

        :param description: ID of scene to retrieve. If None, retrieves a random architecture from the database.
        Defaults to None
        :param whole_scene: if True, return architecture of entire scene, else returns only a single room (chosen at
        random). Defaults to False.
        :return: architecture of room in scene, including walls, floors, and ceilings.
        """
        arch_url = self.__arch_db.get(description)  # assumes description is ID of scene
        print(f"Retrieving architecture from {arch_url}")
        try:
            resp = self.session.get(arch_url)
            arch_json = json.loads(resp.text)
            Architecture.LAST_ARCH = arch_json
        except requests.exceptions.SSLError as e:
            print(f"[ERROR] {e}")
            arch_json = Architecture.LAST_ARCH
        arch = Architecture.from_json(arch_json)
        print(f"Loaded architecture: {arch.id}")

        arch.set_axes(Architecture.DEFAULT_UP, Architecture.DEFAULT_FRONT, invert=True, rotate=np.pi)

        if not whole_scene:
            large_rooms = []

            # filtering rooms with walls, since sometimes the entire scene is considered one "room"
            for room in arch.rooms:
                if room.wall_sides:
                    rooms = []
                    if room.floor is not None:
                        rooms.append(np.array([[p.x, p.y] for p in room.floor.points]))
                    all_pts = np.concatenate(rooms, axis=0)

                    # get bounds and scale points to [pad, room_dim - pad] range
                    min_bound = np.min(all_pts, axis=0, keepdims=True)
                    max_bound = np.max(all_pts, axis=0, keepdims=True)
                    sz = max_bound - min_bound

                    if np.all(sz >= min_size).item():
                        large_rooms.append(room)

            if large_rooms:
                room = random.choice(large_rooms)
                arch = arch.filter_by_rooms(room.id)
                print(f"Selected room: {room.id}")
                arch.center_architecture()
                return arch

            else:  # repeat until you find a large enough room
                print("Could not find a large room in architecture. Repeating search...")
                return self.retrieve(description, whole_scene, min_size)
