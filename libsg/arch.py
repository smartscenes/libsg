"""
arch.py
-------
Implements functionality related to the storing and manipulation of scene architectures (the overall room shape, walls,
floors, and ceilings).
"""

import base64
import copy
import struct
import sys
from itertools import product
from pathlib import Path
from typing import Callable, Self

import numpy as np
import torch
from shapely import Polygon
from shapely.geometry import MultiPoint

from libsg.scene_types import ArchElement, Room, Opening, Wall, Floor, Ceiling, Window, Door, JSONDict


class Architecture:
    """Data structure for scene architectures"""

    DEFAULT_VERSION = "arch@1.0.2"
    DEFAULT_UP = [0, 0, 1]
    DEFAULT_FRONT = [0, 1, 0]
    DEFAULT_COORDS2D = [0, 1]
    DEFAULT_SCALE_TO_METERS = 1
    DEFAULT_ELEM_PARAMS = {
        "Wall": {"depth": 0.1, "extraHeight": 0.035},
        "Ceiling": {"depth": 0.05},
        "Floor": {"depth": 0.05},
    }

    def __init__(self, id, imagedir=None):
        self.id = id
        self.version = Architecture.DEFAULT_VERSION
        self.up = Architecture.DEFAULT_UP
        self.front = Architecture.DEFAULT_FRONT
        self.coords2d = Architecture.DEFAULT_COORDS2D
        self.scale_to_meters = Architecture.DEFAULT_SCALE_TO_METERS
        self.defaults = copy.deepcopy(Architecture.DEFAULT_ELEM_PARAMS)

        self.imagedir = Path(imagedir) if imagedir is not None else None
        self.elements = []
        self.materials = []
        self.textures = []
        self.images = []
        self.is_typed = False

        self.elements: list[ArchElement] = []
        self._rooms: dict[str, Room] = {}
        self.openings: list[Opening] = []

        self.__special_mats = {"Glass": {"uuid": "mat_glass", "color": "#777777", "opacity": 0.1, "transparent": True}}
        self.__included_special_mats = []

    @property
    def rooms(self) -> list[Room]:
        """Returns list of rooms in architecture"""
        return list(self._rooms.values())  # shallow copy

    def ensure_typed(self) -> None:
        """If self.is_typed is False, ensures that all elements are of type ArchElement. This operation should no-op if
        an element is already of type ArchElement.

        Note that this method supports a naive conversion from JSON entry to ArchElement and may not work where some
        information is omitted (e.g. as defaults).
        """
        # Make sure elements have correct type
        if not self.is_typed:
            self.elements = [
                ArchElement.from_json(element) if not isinstance(element, ArchElement) else element
                for element in self.elements
            ]
            self.is_typed = True

    def get_element_by_id(self, id: str) -> ArchElement:
        """Return element with specified ID."""
        return next(self.find_elements(lambda elem: elem.id == id), None)

    def find_elements(self, cond: Callable[[ArchElement], bool]) -> filter:
        """Return elements matching specified condition."""
        return filter(cond, self.elements)

    def find_elements_by_type(self, element_type: str) -> filter:
        """Return all elements of the specified type (e.g. "Wall", "Ceiling", "Floor")."""
        return filter(lambda elem: elem.type == element_type, self.elements)

    def add_element(self, element: JSONDict) -> ArchElement:
        """
        Add element to architecture, parsing its type into the appropriate ArchElement subclass.

        :param element: a JSON dictionary representing the element to be added. Must be of one of the following forms:

        Wall:
            {
                "id": str,
                "type": "Wall",
                "points": [[x, y, z], ...],
                "height": float,  # height of wall
                "depth": float,  # thickness of wall; pulled from default values if not provided
                "materials": [{"diffuse": str, "name": str}],
                "roomId": str,
            }

        Floor:
            {
                "id": str,
                "type": "Floor",
                "points": [[x, y, z], ...],
                "depth": float,  # thickness of floor; pulled from default values if not provided
                "materials": [{"diffuse": str, "name": str}],
                "roomId": str,
            }

        Ceiling:
            {
                "id": str,
                "type": "Ceiling",
                "points": [[x, y, z], ...],
                "depth": float,  # thickness of floor; pulled from default values if not provided
                "materials": [{"diffuse": str, "name": str}],
                "roomId": str,
            }

        This format is the format specified in STK.

        :return: corresponding ArchElement subclass instance
        """
        room_id = element["roomId"]

        if room_id not in self._rooms:
            self._rooms[room_id] = Room(room_id)

        match element["type"]:
            case "Wall":
                element["depth"] = element.get("depth", self.defaults["Wall"]["depth"])
                for i in range(len(element["materials"])):
                    element["materials"][i]["diffuse"] = "#888899"
                elem = Wall.from_json(element)
                if elem.id not in self._rooms[room_id].wall_sides:
                    self._rooms[room_id].wall_sides.append(elem.id)

                for hole in element.get("holes", []):
                    opening_min = hole["box"]["min"]
                    opening_max = hole["box"]["max"]
                    w = opening_max[0] - opening_min[0]
                    m = (opening_min[0] + (w / 2)) / elem.width
                    h = opening_max[1] - opening_min[1]
                    elev = opening_min[1] + h / 2
                    if hole["type"].lower() == "window":
                        opening = Window(
                            id=hole["id"],
                            parent=elem,
                            mid=m,
                            height=h,
                            width=w,
                            elevation=elev,
                        )
                        elem.openings.append(opening)  # Add opening to Wall instance
                    elif hole["type"].lower() == "door":
                        opening = Door(
                            id=hole["id"],
                            parent=elem,
                            mid=m,
                            height=h,
                            width=w,
                            elevation=elev,
                        )
                        elem.openings.append(opening)  # Add opening to Wall instance
                    else:
                        raise ValueError(f"Unsupported hole type: {hole['type']}")
        
            case "Floor":
                for i in range(len(element["materials"])):
                    element["materials"][i]["diffuse"] = "#929898"
                element["depth"] = element.get("depth", self.defaults["Floor"]["depth"])
                elem = Floor.from_json(element)
                self._rooms[room_id].floor = elem

            case "Ceiling":
                element["depth"] = element.get("depth", self.defaults["Ceiling"]["depth"])
                elem = Ceiling.from_json(element)
                self._rooms[room_id].ceiling = elem

            case _:
                raise ValueError(f"Unsupported type: {elem['type']}")

        self.elements.append(elem)
        self.is_typed = True  # FIXME: this bool could become inconsistent with the self.elements
        return elem

    def add_elements(self, element: list[JSONDict]):
        """Add multiple elements to architecture."""
        for elem in element:
            self.add_element(elem)

    def create_material(self, element: JSONDict, name: str, flipY: bool = True):
        id = element["id"]
        imagefile = self.imagedir.joinpath(f"{id}.png")
        assert imagefile.suffix == ".png", "only PNG format supported currently"
        if not imagefile.exists():
            print(f"[Warning] image file {imagefile} not found, skipping texture creation.", file=sys.stderr)
            return
        element["materials"] = [{"name": name, "materialId": f"mat_{id}"}]
        self.materials.append({"uuid": f"mat_{id}", "map": f"tex_{id}"})
        img_bytes = open(imagefile, "rb").read()
        blob = base64.b64encode(img_bytes).decode("ascii")
        width, height = struct.unpack(">LL", img_bytes[16:24])
        self.textures.append(
            {"uuid": f"tex_{id}", "repeat": [100 / width, 100 / height], "image": f"img_{id}", "flipY": flipY}
        )
        self.images.append({"uuid": f"img_{id}", "url": f"data:image/png;base64,{blob}"})

    def set_special_material(self, element: JSONDict, name: str, mat_name: str):
        element["materials"] = [{"name": name, "materialId": f"mat_{mat_name.lower()}"}]
        if mat_name not in self.__included_special_mats:
            self.materials.append(self.__special_mats[mat_name])
            self.__included_special_mats.append(mat_name)

    def populate_materials(self, element: JSONDict):
        etype = element["type"]
        if "materials" in element and isinstance(element["materials"][0], str):
            name = "inside" if etype in ["Wall", "Railing"] else "surface"
            self.set_special_material(element=element, name="inside", mat_name=element["materials"][0])
        elif self.imagedir is not None:
            if etype in ["Wall", "Railing"] and element["height"] > 0:
                self.create_material(element=element, name="inside", flipY=True)
            elif etype in ["Ceiling", "Floor", "Landing", "Ground"]:
                self.create_material(element=element, name="surface", flipY=etype != "Ceiling")
        if "railing" in element:
            for r in element["railing"]:
                self.populate_materials(r)

    def get_room_mask(self, *, layout_size: int = 64, room_dims: list[float], device=None) -> torch.Tensor:
        """Get binary room mask for architecture.

        :param room_dim: size of room mask, defaults to 64
        :param floor_perc: percent of room mask size to correspond to active room area, defaults to 0.8
        :return: tensor mask representing open space in room
        """

        def map_to_mask(pts: np.ndarray, *, layout_size: int = 64, room_dims: list[float]):
            pts_norm = pts / (np.expand_dims(np.array([room_dims[0], room_dims[1]]), axis=0) / 2)  # -1 to 1
            return (pts_norm + 1) * (layout_size / 2)  # 0 to layout_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        room_mask = torch.zeros(size=(1, 1, layout_size, layout_size), dtype=torch.float32, device=device)

        rooms = []
        for room in self.rooms:
            if room.floor is not None:
                rooms.append(np.array([[p.x, p.y] for p in room.floor.points]))
        all_pts = np.concatenate(rooms, axis=0)

        # get bounds and scale points to [pad, room_dim - pad] range
        min_bound = np.min(all_pts, axis=0, keepdims=True)
        max_bound = np.max(all_pts, axis=0, keepdims=True)
        print(f"{min_bound=}, {max_bound=}")

        # iterate through rooms and white-list floors
        for room in self.rooms:
            if room.floor is not None:
                pts = np.array([[p.x, p.y] for p in room.floor.points])
                pts_scaled = map_to_mask(pts, layout_size=layout_size, room_dims=room_dims)

                # apply room to mask
                room_poly = Polygon(pts_scaled)
                lattice = MultiPoint(list(product(np.arange(layout_size), np.arange(layout_size))))

                intersection = lattice.intersection(room_poly)
                x_coords = []
                y_coords = []
                for pt in intersection.geoms:
                    x_coords.append(int(pt.x))
                    y_coords.append(int(pt.y))
                room_mask[0, 0, np.array(y_coords), np.array(x_coords)] = 1.0

        # re-mask out walls
        for elem in self.elements:
            if elem.type == "Wall":
                wall_bbox = elem.bbox3D
                wall_bounds = np.array([[wall_bbox.min.x, wall_bbox.min.y], [wall_bbox.max.x, wall_bbox.max.y]])
                wall_bounds_rescaled = map_to_mask(wall_bounds, layout_size=layout_size, room_dims=room_dims).astype(
                    int
                )
                room_mask[
                    0,
                    0,
                    np.min(wall_bounds_rescaled[:, 1]) : np.max(wall_bounds_rescaled[:, 1]) + 1,
                    np.min(wall_bounds_rescaled[:, 0]) : np.max(wall_bounds_rescaled[:, 0]) + 1,
                ] = 0.0
        
        open_indices = np.where(room_mask == 1)
        min_pixels = np.array([np.min(open_indices[3]), np.min(open_indices[2])])
        max_pixels = np.array([np.max(open_indices[3]), np.max(open_indices[2])])
        print("pixel bounds", min_pixels, max_pixels)

        return room_mask

    def set_axes(self, up: list[int], front: list[int], invert: bool, rotate: float) -> Self:
        """Set axes for architecture, if different from original axes.

        This method updates all elements in the architecture to reflect the new axes and also updates the `up` and
        `front` properties with the new axis directions.

        :param up: new upward-facing axis, in the form of a one-hot vector. This method does NOT support arbitrary axis
        directions.
        :param front: new front-facing axis, in the form of a one-hot vector. This method does NOT support arbitrary
        axis directions.
        :param invert: if True, invert the front-facing axis to switch from LHS to RHS (or vice versa)
        :param rotate: rotate axes about upward-facing axis by specified angle (in radians)
        :return: self
        """

        def get_axis_index(axis_vec):
            """Get index of axis.

            Example:
            >>> get_axis_index([0, 1, 0]) == 1
            """
            return np.array([0, 1, 2])[list(map(bool, axis_vec))].item()

        orig_up = get_axis_index(self.up)
        orig_front = get_axis_index(self.front)
        target_up = get_axis_index(up)
        target_front = get_axis_index(front)

        if orig_up == target_up and orig_front == target_front:
            return self

        for elem in self.elements:
            new_points = []
            for point in elem.points:
                # swap axes
                point = point.swap_axes(orig_up, target_up)
                if orig_front == target_up:  # need to update orig_front if affected by initial swap
                    orig_front = orig_up
                point = point.swap_axes(orig_front, target_front)

                # invert one axis to switch from LH to RH (or vice versa)
                if invert:
                    point = point.invert(target_front)

                # rotate about axis
                if rotate != 0:
                    point = point.rotate(target_up, rotate)
                new_points.append(point)
            elem.points = new_points

        self.up = up
        self.front = front
        return self

    def filter_by_rooms(self, room_ids: list[str]) -> Self:
        """Return new architecture with only specified room_ids."""
        arch = Architecture(self.id)
        arch.version = self.version
        arch.up = self.up
        arch.front = self.front
        arch.unit = self.scale_to_meters
        arch._rooms = {room.id: room for room in self.rooms if room.id in room_ids}

        arch.add_elements([elem.to_json() for elem in self.elements if elem.room_id in room_ids])

        # TODO: do these need to be filtered?
        arch.materials = self.materials
        arch.textures = self.textures
        arch.images = self.images
        return arch

    def center_architecture(self) -> Self:
        """
        Modifies the architecture to center it in the coordinate system.

        :return: self
        """
        # compute centroid
        rooms = []
        for room in self.rooms:
            if room.floor is not None:
                rooms.append(np.array([[p.x, p.y, 0] for p in room.floor.points]))
        all_pts = np.concatenate(rooms, axis=0)
        min_bound = np.min(all_pts, axis=0)
        max_bound = np.max(all_pts, axis=0)
        centroid = (min_bound + max_bound) / 2

        for elem in self.elements:
            elem.translate(-centroid)

        return self

    def to_json(self) -> JSONDict:
        """Convert architecture to JSON form."""
        return {
            "version": self.version,
            "up": self.up,
            "front": self.front,
            "coords2d": self.coords2d,
            "scaleToMeters": self.scale_to_meters,
            "defaults": self.defaults,
            "id": self.id,
            "elements": [elem.to_json() for elem in self.elements] if self.is_typed else self.elements,
            "holes": [elem.to_json() for elem in self.openings], 
            "regions": [elem.to_json() for elem in self.rooms],
            "materials": self.materials,
            "textures": self.textures,
            "images": self.images,
        }

    @classmethod
    def from_json(cls, obj: JSONDict) -> Self:
        """Return Architecture object based on JSON input."""
        arch = Architecture(obj["id"])
        arch.version = obj["version"]
        arch.up = obj["up"]
        arch.front = obj["front"]
        arch.unit = obj["scaleToMeters"]
        arch._rooms = {region["id"]: Room.from_json(region) for region in obj.get("regions", [])}

        arch.add_elements(obj["elements"])  # included openings
        arch.materials = obj.get("materials", [])
        arch.textures = obj.get("textures", [])
        arch.images = obj.get("images", [])
        return arch
