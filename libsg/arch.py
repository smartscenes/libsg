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
import uuid
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Callable, Self

import numpy as np
import torch
from shapely import Polygon
from shapely.geometry import MultiPoint

from libsg.scene_types import ArchElement, Room, Opening, Wall, Floor, Ceiling, Window, Door, JSONDict, Point3D
from libsg.model.holodeck.materials import MaterialsDB


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
        self.raw = None

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

    def add_region(self, region: JSONDict) -> ArchElement:
        room = Room.from_json(region)
        self._rooms[room.id] = room
        return room

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

        open_indices = torch.where(room_mask == 1)[0].cpu().numpy()
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
        arch.defaults = self.defaults
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

    def get_floor_plan(self) -> tuple[list[Point3D], np.ndarray, str]:
        floor_points = []
        floor_elements = list(self.find_elements_by_type("Floor"))
        for floor in floor_elements:
            floor_points.extend([(point[0], point[1], point[2]) for point in floor.points])
        
        # Create faces as triangles
        floor_faces = []
        for i in range(1, len(floor.points) - 1):
            floor_faces.append([0, i, i + 1])
        floor_faces = np.array(floor_faces)
        
        return floor_points, floor_faces, floor.room_id
    
    def get_door_positions(self, room_id):
        door_positions = []
        for element in self.elements:
            if isinstance(element, Wall) and element.room_id == room_id:
                for opening in element.openings:
                    if isinstance(opening, Door):
                        # Get wall start and end points
                        wall_start = np.array([element.points[0].x, element.points[0].y, element.points[0].z])

                        wall_end = np.array([element.points[1].x, element.points[1].y, element.points[1].z])
                        print("wall_start", "wall_end")
                        print(wall_start, wall_end)
                        # Calculate door position along the wall
                        door_position = wall_start + (wall_end - wall_start) * opening.mid

                        door_positions.append(door_position)

        return np.array(door_positions) if door_positions else None

    @classmethod
    def from_json(cls, obj: JSONDict) -> Self:
        """Return Architecture object based on JSON input."""
        arch = Architecture(obj["id"])
        arch.version = obj["version"]
        arch.up = obj["up"]
        arch.front = obj["front"]
        if "defaults" in obj:
            arch.defaults = obj["defaults"]
        arch.unit = obj["scaleToMeters"]
        for region in obj.get("regions", []):
            arch.add_region(region)
        arch._rooms = {region["id"]: Room.from_json(region) for region in obj.get("regions", [])}

        arch.add_elements(obj["elements"])  # included openings
        arch.materials = obj.get("materials", [])
        arch.textures = obj.get("textures", [])
        arch.images = obj.get("images", [])
        arch.raw = obj
        return arch

    @classmethod
    def from_ai2thor(cls, scene: JSONDict, materials: MaterialsDB) -> Self:
        """
        Return Architecture object based on JSON ai2thor input format (e.g. used by Holodeck).

        The main architectural components we need to convert are rooms, walls, floors, ceilings, and openings (doors
        and windows). Holodeck organizes the scene into the following format (ignoring irrelevant fields):
        {
            "doors": [{
                "id": door|idx|room0|room1,
                "openable": bool,
                "openness": 0 or 1,  # 0 for closed, 1 for open
                "room0": room_id,
                "room1": room_id,
                "wall0": wall_id,  # wall for room 0
                "wall1": wall_id,  # wall for room 1
                "holePolygon": [
                    {"x": min_coord_along_wall_dim, "y": 0, "z": 0},  # all doors start at top of floor
                    {"x": max_coord_along_wall_dim, "y": door_height, "z": 0}
                ],
                "assetPosition": {"x": midpoint_along_wall_dim, midpoint_along_height, "z": 0},
                "doorBoxes": [[pt1+perp_vec, pt2+perp_vec, pt2, pt1], [pt1, pt2, pt2-perp_vec, pt1-perp_vec]],  # box containing the door, using a default width of 1.0
                "doorSegment": [
                    [start_x_from_wall_start, start_z_from_wall_start],
                    [end_x_from_wall_start, end_z_from_wall_start]
                ],  # each point pt1, pt2 is wall_coord_start + normalized_vector * {door_start, door_end}
            }],
            "rooms": [...],
            "walls": [{
                "id": wall|roomId|direction|idx(|exterior),
                "roomId": room_id,
                "material": {"name": wall_material},
                "polygon: [
                    {"x": x0, "y": 0, "z": z0},
                    {"x": x0, "y": height, "z": z0},
                    {"x": x1, "y": height, "z": z1},  # either x0 == x1 or z0 == z1
                    {"x": x1, "y": 0, "z": z1}
                ],
                "connected_rooms": [{
                    "intersection": [{"x": x0, "y": 0, "z": z0}, {"x": x1, "y": 0, "z": z1}],
                    "line0": [{"x": x0, "y": 0, "z": z0}, {"x": x1, "y": 0, "z": z1}],  # room0
                    "line1": [{"x": x1, "y": 0, "z": z1}, {"x": x0, "y": 0, "z": z0}],  # room1
                    "roomId": room1_id,  # connected room
                    "wallId": wall_id,  # pair wall of other room
                }],  # empty list if no connected rooms
                "width": float,  # width of wall
                "height": float,  # height of wall
                "direction": "north"|"south"|"east"|"west",  # x0 == x1 for east/west, z0 == z1 for north/south
                "segment": [[x0, z0], [x1, z1]],
                "connect_exterior": exterior_wall_id,  # not present if wall does not connect to exterior
            }],
            "wall_height": float,
            "windows": [{
                "id": door|idx|room0|room1,
                "room0": room_id,
                "room1": room_id,
                "wall0": wall_id,  # wall for room 0
                "wall1": wall_id,  # wall for room 1
                "holePolygon": [
                    {"x": min_coord_along_wall_dim, "y": window_bottom, "z": 0},
                    {"x": max_coord_along_wall_dim, "y": window_top, "z": 0}
                ],
                "assetPosition": {"x": midpoint_along_wall_dim, midpoint_along_height, "z": 0},
                "windowBoxes": [[pt1+perp_vec, pt2+perp_vec, pt2, pt1], [pt1, pt2, pt2-perp_vec, pt1-perp_vec]],  # box containing the window, using a default width of 1.0
                "windowSegment": [
                    [start_x_from_wall_start, start_z_from_wall_start],
                    [end_x_from_wall_start, end_z_from_wall_start]
                ],  # each point pt1, pt2 is wall_coord_start + normalized_vector * {door_start, door_end}
            }],
            "room_pairs": [...],
        }
        """
        arch = Architecture(scene.get("id", str(uuid.uuid4())))
        arch.up = [0, 1, 0]
        arch.front = [0, 0, 1]
        arch.unit = 1.0
        # ProcTHOR default is 0.1/2 but that causes issues with pictures on walls; use half that
        wall_thickness = 0.025
        room_heights = {}

        # doors & windows
        wall_to_holes = {}
        for opening in scene["doors"] + scene["windows"]:
            pts = opening["holePolygon"]
            box = {"min": [pts[0]["x"], pts[0]["y"]], "max": [pts[1]["x"], pts[1]["y"]]}
            hole = {
                "id": f"{opening['id']}",
                "type": "door" if opening["id"].startswith("door") else "window",
                "box": box,
                "wall1": opening["wall1"],
                "asset": {"id": opening["assetId"]},
            }
            wall0_holes = wall_to_holes.get(opening["wall0"], [])
            wall0_holes.append(hole)
            wall1_holes = wall_to_holes.get(opening["wall1"], [])
            wall1_holes.append(copy.deepcopy(hole))
            wall_to_holes[opening["wall0"]] = wall0_holes
            wall_to_holes[opening["wall1"]] = wall1_holes

        # walls
        for wall in scene["walls"]:
            if wall.get("empty", False):  # this is a "non-wall" (boundary between two rooms with no wall)
                continue
            pts = wall["polygon"]  # four points
            # i = (
            #     2 if "exterior" in wall["id"] else 0
            # )  # low points are first two for interior, last two for exterior walls
            pts_low = [[pts[0]["x"], pts[0]["y"], pts[0]["z"]], [pts[3]["x"], pts[3]["y"], pts[3]["z"]]]
            min_y = min(pts[0]["y"], pts[1]["y"], pts[2]["y"], pts[3]["y"])
            max_y = max(pts[0]["y"], pts[1]["y"], pts[2]["y"], pts[3]["y"])
            height = max_y - min_y
            if height >= room_heights.get(wall["roomId"], 0):
                room_heights[wall["roomId"]] = height

            wall_element = {
                "id": wall["id"],
                "type": "Wall",
                "points": pts_low,
                "height": height,
                "roomId": wall["roomId"],
                "depth": wall_thickness,
                "materials": [materials.stk_mat(wall["material"])],
            }

            if wall["id"] in wall_to_holes:  # add any opening holes
                wall_holes = wall_to_holes[wall["id"]]
                for hole in wall_holes:
                    if wall["id"] == hole["wall1"]:  # hole box definition flipped along x for "back wall"
                        d0 = pts_low[1][0] - pts_low[0][0]
                        d1 = pts_low[1][2] - pts_low[0][2]
                        wall_width = np.sqrt(d0 * d0 + d1 * d1)
                        x0 = hole["box"]["min"][0]
                        x1 = hole["box"]["max"][0]
                        hole["box"]["max"][0] = wall_width - x0
                        hole["box"]["min"][0] = wall_width - x1
                wall_element["holes"] = wall_holes

            arch.add_element(wall_element)

        ceiling_mat = copy.copy(scene["proceduralParameters"]["ceilingMaterial"])
        if "ceilingColor" in scene["proceduralParameters"]:
            ceiling_mat["color"] = scene["proceduralParameters"]["ceilingColor"]

        # rooms
        for room in scene["rooms"]:
            poly_pts = [[p["x"], p["y"], p["z"]] for p in room["floorPolygon"]]
            arch.add_region({"id": room["id"]})
            floor = {
                "id": f"{room['id']}|f",
                "type": "Floor",
                "points": poly_pts,
                "roomId": room["id"],
                "materials": [materials.stk_mat(room["floorMaterial"])],
            }
            arch.add_element(floor)

            room_height = room_heights[room["id"]]
            ceil_poly_pts = [[p["x"], p["y"] + room_height, p["z"]] for p in room["floorPolygon"]]
            ceiling = {
                "id": f"{room['id']}|c",
                "type": "Ceiling",
                "points": ceil_poly_pts,
                "roomId": room["id"],
                "materials": [materials.stk_mat(ceiling_mat)],
            }
            arch.add_element(ceiling)

        import pprint

        print("RAW:")
        print("----------------")
        pprint.pprint(scene)
        print("BEFORE:")
        print("----------------")
        pprint.pprint(arch.to_json())
        # arch.center_architecture()
        arch.set_axes(Architecture.DEFAULT_UP, Architecture.DEFAULT_FRONT, invert=True, rotate=np.pi)
        arch.raw = scene
        print("AFTER:")
        print("----------------")
        pprint.pprint(arch.to_json())

        return arch

    # @classmethod
    # def from_holodeck(cls, scene: JSONDict) -> Self:
    #     """
    #     Return Architecture object based on JSON Holodeck input format.

    #     The main architectural components we need to convert are rooms, walls, floors, ceilings, and openings (doors
    #     and windows). Holodeck organizes the scene into the following format (ignoring irrelevant fields):
    #     {
    #         "doors": [{
    #             "id": door|idx|room0|room1,
    #             "openable": bool,
    #             "openness": 0 or 1,  # 0 for closed, 1 for open
    #             "room0": room_id,
    #             "room1": room_id,
    #             "wall0": wall_id,  # wall for room 0
    #             "wall1": wall_id,  # wall for room 1
    #             "holePolygon": [
    #                 {"x": min_coord_along_wall_dim, "y": 0, "z": 0},  # all doors start at top of floor
    #                 {"x": max_coord_along_wall_dim, "y": door_height, "z": 0}
    #             ],
    #             "assetPosition": {"x": midpoint_along_wall_dim, midpoint_along_height, "z": 0},
    #             "doorBoxes": [[pt1+perp_vec, pt2+perp_vec, pt2, pt1], [pt1, pt2, pt2-perp_vec, pt1-perp_vec]],  # box containing the door, using a default width of 1.0
    #             "doorSegment": [
    #                 [start_x_from_wall_start, start_z_from_wall_start],
    #                 [end_x_from_wall_start, end_z_from_wall_start]
    #             ],  # each point pt1, pt2 is wall_coord_start + normalized_vector * {door_start, door_end}
    #         }],
    #         "rooms": [...],
    #         "walls": [{
    #             "id": wall|roomId|direction|idx(|exterior),
    #             "roomId": room_id,
    #             "material": {"name": wall_material},
    #             "polygon: [
    #                 {"x": x0, "y": 0, "z": z0},
    #                 {"x": x0, "y": height, "z": z0},
    #                 {"x": x1, "y": height, "z": z1},  # either x0 == x1 or z0 == z1
    #                 {"x": x1, "y": 0, "z": z1}
    #             ],
    #             "connected_rooms": [{
    #                 "intersection": [{"x": x0, "y": 0, "z": z0}, {"x": x1, "y": 0, "z": z1}],
    #                 "line0": [{"x": x0, "y": 0, "z": z0}, {"x": x1, "y": 0, "z": z1}],  # room0
    #                 "line1": [{"x": x1, "y": 0, "z": z1}, {"x": x0, "y": 0, "z": z0}],  # room1
    #                 "roomId": room1_id,  # connected room
    #                 "wallId": wall_id,  # pair wall of other room
    #             }],  # empty list if no connected rooms
    #             "width": float,  # width of wall
    #             "height": float,  # height of wall
    #             "direction": "north"|"south"|"east"|"west",  # x0 == x1 for east/west, z0 == z1 for north/south
    #             "segment": [[x0, z0], [x1, z1]],
    #             "connect_exterior": exterior_wall_id,  # not present if wall does not connect to exterior
    #         }],
    #         "wall_height": float,
    #         "windows": [{
    #             "id": door|idx|room0|room1,
    #             "room0": room_id,
    #             "room1": room_id,
    #             "wall0": wall_id,  # wall for room 0
    #             "wall1": wall_id,  # wall for room 1
    #             "holePolygon": [
    #                 {"x": min_coord_along_wall_dim, "y": window_bottom, "z": 0},
    #                 {"x": max_coord_along_wall_dim, "y": window_top, "z": 0}
    #             ],
    #             "assetPosition": {"x": midpoint_along_wall_dim, midpoint_along_height, "z": 0},
    #             "windowBoxes": [[pt1+perp_vec, pt2+perp_vec, pt2, pt1], [pt1, pt2, pt2-perp_vec, pt1-perp_vec]],  # box containing the window, using a default width of 1.0
    #             "windowSegment": [
    #                 [start_x_from_wall_start, start_z_from_wall_start],
    #                 [end_x_from_wall_start, end_z_from_wall_start]
    #             ],  # each point pt1, pt2 is wall_coord_start + normalized_vector * {door_start, door_end}
    #         }],
    #         "room_pairs": [...],
    #     }
    #     """
    #     # generate JSON form
    #     arch_stk = {
    #         "id": str(uuid.uuid4()),
    #         "regions": [],
    #         "elements": [],
    #         "up": [0, 1, 0],
    #         "front": [0, 0, 1],
    #         "scaleToMeters": Architecture.DEFAULT_SCALE_TO_METERS,
    #         "version": Architecture.DEFAULT_VERSION,
    #     }

    #     # parse walls as a lookup
    #     walls_raw = {wall["id"]: wall for wall in scene["walls"]}

    #     # parse openings
    #     holes_lookup = defaultdict(list)
    #     opening_lookup = {
    #         "doors": {"type": "Door"},
    #         "windows": {"type": "Window"},
    #     }
    #     for opening_type, opening_labels in opening_lookup.items():
    #         for opening in scene[opening_type]:
    #             wall_0 = walls_raw[opening["wall0"]]
    #             wall_1 = walls_raw[opening["wall1"]]

    #             for wall in (wall_0, wall_1):
    #                 x_coord = [pt[0] for pt in wall["segment"]]
    #                 z_coord = [pt[1] for pt in wall["segment"]]
    #                 delta = x_coord if x_coord[1] - x_coord[0] != 0 else z_coord
    #                 height = [pt["y"] for pt in opening["holePolygon"]]  # [0.0, height]

    #                 min_dist = min(
    #                     abs(opening["holePolygon"][0]["x"] - delta[0]), abs(opening["holePolygon"][1]["x"] - delta[0])
    #                 )
    #                 max_dist = max(
    #                     abs(opening["holePolygon"][0]["x"] - delta[0]), abs(opening["holePolygon"][1]["x"] - delta[0])
    #                 )
    #                 height = [corner["y"] for corner in opening["holePolygon"]]  # assume sorted [0.0, height]
    #                 holes_lookup[wall["id"]].append(
    #                     {
    #                         "id": opening["id"],
    #                         "type": opening_labels["type"],
    #                         "box": {"min": [min_dist, height[0]], "max": [max_dist, height[1]]},
    #                     }
    #                 )
    #     breakpoint()

    #     # parse walls and openings
    #     rooms_to_walls = defaultdict(list)
    #     walls = []
    #     for idx, wall in enumerate(scene["walls"]):
    #         wall_id = f"wall_{idx}"
    #         # Note that every wall occurs in pairs, one on the inside and one on the outside. In STK, these are treated
    #         # as one wall, so we need to combine them in Holodeck.
    #         if "exterior" in wall["roomId"]:
    #             continue

    #         # track walls per room
    #         rooms_to_walls[wall["roomId"]].append(wall_id)

    #         # shift wall to account for wall thickness, since Holodeck assumes 0-thickness walls
    #         # wall_depth = Architecture.DEFAULT_ELEM_PARAMS["Wall"]["depth"]
    #         wall_depth = 0.1  # TODO: need to account for wall depth later on
    #         x_coord = [pt[0] for pt in wall["segment"]]
    #         z_coord = [pt[1] for pt in wall["segment"]]
    #         wall_points = np.array([[x, 0.0, z] for x, z in zip(x_coord, z_coord)])

    #         # add wall
    #         walls.append(
    #             {
    #                 "id": wall["id"],
    #                 "type": "Wall",
    #                 "points": wall_points.tolist(),
    #                 "depth": wall_depth,
    #                 "height": wall["height"],
    #                 "holes": holes_lookup[wall["id"]],
    #                 "materials": [
    #                     {"diffuse": "#888899", "name": "inside", "texture": "painted_white_plane_17521"},
    #                     {"diffuse": "#888899", "name": "outside", "texture": "painted_white_plane_17521"},
    #                 ],  # TODO: populate from Holodeck
    #                 "roomId": wall["roomId"],
    #             }
    #         )

    #     for room in scene["rooms"]:
    #         room_id = room["id"]

    #         # parse floor
    #         arch_stk["elements"].append(
    #             {
    #                 "id": f"{room_id}|floor",
    #                 "type": "Floor",
    #                 "points": [[pt["x"], pt["y"], pt["z"]] for pt in room["floorPolygon"]],
    #                 "roomId": room_id,
    #                 "depth": Architecture.DEFAULT_ELEM_PARAMS["Floor"]["depth"],
    #                 "materials": [{"diffuse": "#929898", "name": "surface", "texture": "wood_cream_plane_1375"}],
    #             }
    #         )

    #         # parse ceiling
    #         arch_stk["elements"].append(
    #             {
    #                 "id": f"{room_id}|ceiling",
    #                 "type": "Ceiling",
    #                 "points": [[pt["x"], pt["y"], pt["z"]] for pt in room["floorPolygon"]],
    #                 "roomId": room_id,
    #                 "depth": Architecture.DEFAULT_ELEM_PARAMS["Ceiling"]["depth"],
    #                 "materials": [{"diffuse": "#ffffff", "name": "surface", "texture": "painted_white_plane_17521"}],
    #             }
    #         )

    #         # add region
    #         arch_stk["regions"].append(
    #             {
    #                 "id": room_id,
    #                 "wallIds": rooms_to_walls[room["id"]],
    #                 "type": room["roomType"],
    #             }
    #         )

    #     arch_stk["elements"].extend(walls)

    #     # instantiate architecture
    #     arch = Architecture.from_json(arch_stk)
    #     arch.set_axes(Architecture.DEFAULT_UP, Architecture.DEFAULT_FRONT, invert=True, rotate=np.pi)
    #     arch.center_architecture()
    #     arch.raw = scene
    #     return arch
