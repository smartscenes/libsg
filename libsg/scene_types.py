"""
scene_types.py
---
Definitions of an assortment of utility classes to represent geometric objects (e.g. points, boxes), architecture
elements of a scene, and specifications for API requests and aspects of scenes.
"""

import inspect
import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Any, Optional, Self, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from shapely import Polygon
from shapely.geometry import MultiPoint

JSONDict = dict[str, Any]


class Point:
    """Generic point class in multiple dimensions"""

    SIZE = 0

    def length_sq(self_or_cls, p: Optional[Sequence[float] | Self] = None) -> float:
        """Compute squared length of vector from origin to point"""
        if inspect.isclass(self_or_cls):
            cls = self_or_cls
        else:
            cls = type(self_or_cls)
            p = self_or_cls
        return sum([p[i] * p[i] for i in range(0, cls.SIZE)])

    def length(self_or_cls, p: Optional[Sequence[float] | Self] = None) -> float:
        """Compute length of vector from origin to point"""
        return math.sqrt(self_or_cls.length_sq(p))

    @classmethod
    def min(cls, points: Sequence[Self]) -> Self:
        """Return element-wise minimum of collection of points"""
        return cls(*[min([p[i] for p in points]) for i in range(0, cls.SIZE)])

    @classmethod
    def max(cls, points: Sequence[Self]) -> Self:
        """Return element-wise maximum of collection of points"""
        return cls(*[max([p[i] for p in points]) for i in range(0, cls.SIZE)])

    @classmethod
    def sum(cls, points: Sequence[Self]) -> Self:
        """Return element-wise sum for collection of points"""
        return cls(*[sum([p[i] for p in points]) for i in range(0, cls.SIZE)])

    @classmethod
    def mean(cls, points: Sequence[Self]) -> Self:
        """Return mean/centroid of collection of points"""
        k = len(points)
        return cls(*[sum([p[i] for p in points]) / k for i in range(0, cls.SIZE)])

    @classmethod
    def sub(cls, a: Self, b: Self) -> Self:
        """Subtract one point from another element-wise"""
        return cls(*[(a[i] - b[i]) for i in range(0, cls.SIZE)])

    def __sub__(self, other: Self) -> Self:
        return self.__class__.sub(self, other)

    @classmethod
    def add(cls, a: Self, b: Self) -> Self:
        """Add two points element-wise"""
        return cls(*[(a[i] + b[i]) for i in range(0, cls.SIZE)])

    def __add__(self, other: Self) -> Self:
        return self.__class__.add(self, other)

    @classmethod
    def mult(cls, a: Self, b: float) -> Self:
        """Multiply point by scalar"""
        return cls(*[a[i] * b for i in range(0, cls.SIZE)])

    def __mul__(self, other: float) -> Self:
        return self.__class__.mult(self, other)

    @classmethod
    def distance_sq(cls, a: Self, b: Self) -> float:
        """Compute squared distance between two points"""
        return sum([(a[i] - b[i]) * (a[i] - b[i]) for i in range(0, cls.SIZE)])

    @classmethod
    def distance(cls, a: Self, b: Self) -> float:
        """Compute distance between two points"""
        return math.sqrt(cls.distance_sq(a, b))

    @classmethod
    def fromlist(cls, l: Sequence[float]) -> Self:
        """Return Point from list of coordinates"""
        return cls(*[p for p in l])

    def swap_axes(self, axis_1: int, axis_2: int) -> Self:
        """Return a new point with two axes swapped.

        Example:
        >>> point = Point3D(1, 2, 3)
        >>> point.swap_axes(0, 1) == Point3D(2, 1, 3)

        :param axis_1: index of first axis to swap
        :param axis_2: index of second axis to swap
        :return: new point with axes swapped.
        """
        if axis_1 == axis_2:
            return self

        point = self.tolist()
        transform_matrix = np.eye(len(point))
        row_1 = np.copy(transform_matrix[axis_1])
        row_2 = np.copy(transform_matrix[axis_2])
        transform_matrix[axis_1, :] = row_2
        transform_matrix[axis_2, :] = row_1

        new_point = transform_matrix @ np.array(point)

        return self.__class__.fromlist(new_point)

    def invert(self, axis: int) -> Self:
        """Returns new point with specified axis inverted (negated).

        Example:
        >>> point = Point3D(1, 2, 3)
        >>> point.invert(0) == Point3D(-1, 2, 3)

        :param axis: index of axis to invert
        :return: new point with axis inverted
        """
        point = self.tolist()
        point[axis] = -point[axis]
        return self.__class__.fromlist(point)

    def __eq__(self, other: Self) -> bool:
        return self.tolist() == other.tolist()

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + ", ".join(str(self[i]) for i in range(self.SIZE)) + ")"


class Point2D(Point):
    """2D point class."""

    SIZE = 2

    def __init__(self, x: float, y: float):
        """
        :param x: x-coordinate of point
        :param y: y-coordinate of point
        """
        super().__init__()
        self.x: float = x
        self.y: float = y

    def __str__(self):
        return str(self.tolist())

    def __getitem__(self, key: int) -> float:
        return (self.x, self.y)[key]

    def tolist(self) -> list[float]:
        """Return point as a list of coordinates"""
        return [self.x, self.y]


class Point3D(Point):
    """3D point class."""

    SIZE = 3

    def __init__(self, x: float, y: float, z: float):
        """
        :param x: x-coordinate of point
        :param y: y-coordinate of point
        :param z: z-coordinate of point
        """
        super().__init__()
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def __str__(self) -> str:
        return str(self.tolist())

    def __getitem__(self, key: int) -> float:
        return (self.x, self.y, self.z)[key]

    def tolist(self) -> list[float]:
        """Return point as a list of coordinates"""
        return [self.x, self.y, self.z]

    def scale(self, s: float) -> Self:
        """Scale point along vector with origin by a factor of s."""
        self.x = self.x * s
        self.y = self.y * s
        self.z = self.z * s
        return self

    def normalize(self) -> Self:
        """Normalize vector from origin to point."""
        length = self.length()
        self.scale(1 / length)
        return self

    def rotate(self, axis: int, angle: float) -> Self:
        """Rotate a point about an axis index by a given angle (in radians)."""
        point = self.tolist()
        rotvec = np.zeros((3,), dtype=float)
        rotvec[axis] = angle
        new_point = Rotation.from_rotvec(rotvec).as_matrix() @ np.array(point)
        return self.__class__.fromlist(new_point)

    def translate(self, vec: np.array) -> Self:
        point = np.array(self.tolist())
        transformed = point + vec
        return self.__class__.fromlist(transformed)


class BBox2D:
    """Bounding box in 2 dimensions."""

    def __init__(self):
        """
        :param min: minimum coordinate per axis of the bounding box
        :param max: maximum coordinate per axis of the bounding box
        """
        self.min: Point2D
        self.max: Point2D

    @classmethod
    def from_point_list(cls, points):
        """Construct 2D bounding box from list of points"""
        min_point = Point2D.min(points)
        max_point = Point2D.max(points)
        return BBox2D(min_point, max_point)


class BBox3D:
    """Axis-aligned bounding box in 3 dimensions."""

    LEFT = 0
    RIGHT = 1
    BOTTOM = 2
    TOP = 3
    FRONT = 4
    BACK = 5

    def __init__(self, min, max):
        """
        :param min: minimum coordinates for bounding box
        :param max: maximum coordinates for bounding box
        """
        self.min: Point3D = min
        self.max: Point3D = max

    @property
    def centroid(self) -> Point3D:
        """Compute centroid of bounding box"""
        mid = Point3D.mean([self.min, self.max])
        return mid

    @property
    def dims(self) -> Point3D:
        """Compute box dimensions of bounding box"""
        point = Point3D.sub(self.max, self.min)
        return point

    def __str__(self) -> str:
        return f"BBox3D({self.min}, {self.max})"

    def get_face_center(self, index: int) -> Point3D:
        """Get coordinate of center of bounding box face"""
        point = self.centroid
        if index == BBox3D.LEFT:
            point.x = self.min.x
        elif index == BBox3D.RIGHT:
            point.x = self.max.x
        elif index == BBox3D.BOTTOM:
            point.z = self.min.z
        elif index == BBox3D.TOP:
            point.z = self.max.z
        elif index == BBox3D.FRONT:
            point.y = self.max.y
        elif index == BBox3D.BACK:
            point.y = self.min.y
        return point

    def get_face_outnormal(self, index: int) -> Point3D:
        """Get unit vector normal to specified face"""
        if index == BBox3D.LEFT:
            return Point3D(-1, 0, 0)
        elif index == BBox3D.RIGHT:
            return Point3D(+1, 0, 0)
        elif index == BBox3D.BOTTOM:
            return Point3D(0, 0, -1)
        elif index == BBox3D.TOP:
            return Point3D(0, 0, +1)
        elif index == BBox3D.FRONT:
            return Point3D(0, -1, 0)
        elif index == BBox3D.BACK:
            return Point3D(0, +1, 0)

    def get_face_dims(self, index: int) -> Point3D:
        """Get dimensions of specified face"""
        dims = self.dims
        if index == BBox3D.LEFT or index == BBox3D.RIGHT:
            return Point3D(dims.y, dims.z, dims.x)
        elif index == BBox3D.BOTTOM or index == BBox3D.TOP:
            return Point3D(dims.x, dims.y, dims.z)
        else:
            return Point3D(dims.x, dims.z, dims.y)

    def get_point_on_face(self, index: int, r: list[float], margin: Optional[tuple[float, float]]) -> Point3D:
        face_dims = self.get_face_dims(index)
        p0 = self.min
        p1 = self.max
        m0 = margin[0] if margin else 0
        m1 = margin[1] if margin else 0
        d0 = r[0] * (face_dims[0] - m0) + m0
        d1 = r[1] * (face_dims[1] - m1) + m1

        if index == BBox3D.LEFT:
            point = Point3D(p0[0], p0[1] + d0, p0[2] + d1)
        elif index == BBox3D.RIGHT:
            point = Point3D(p1[0], p0[1] + d0, p0[2] + d1)
        elif index == BBox3D.BOTTOM:
            point = Point3D(p0[0] + d0, p0[1] + d1, p0[2])
        elif index == BBox3D.TOP:
            point = Point3D(p0[0] + d0, p0[1] + d1, p1[2])
        elif index == BBox3D.FRONT:
            point = Point3D(p0[0] + d0, p0[1], p0[2] + d1)
        elif index == BBox3D.BACK:
            point = Point3D(p0[0] + d0, p1[1], p0[2] + d1)
        return point

    def sample_face(self, index: int, margin: Optional[tuple[float, float]]) -> Point3D:
        """Sample a random point on the face of the bounding box"""
        return self.get_point_on_face(index, [random.uniform(0, 1), random.uniform(0, 1)], margin)

    def contains(self, p: Point3D) -> bool:
        """Returns True if point is within box bounds (inclusive)"""
        return (
            (self.max.x >= p.x >= self.min.x)
            and (self.max.y >= p.y >= self.min.y)
            and (self.max.z >= p.z >= self.min.z)
        )

    @classmethod
    def from_point_list(cls, points) -> Self:
        """Construct axis-aligned bounding box containing all of a collection of points"""
        min_point = Point3D.min(points)
        max_point = Point3D.max(points)
        return BBox3D(min_point, max_point)

    @classmethod
    def from_min_max(cls, min, max) -> Self:
        """Construct axis-aligned bounding box given min and max coordinates"""
        min_point = Point3D.fromlist(min)
        max_point = Point3D.fromlist(max)
        return BBox3D(min_point, max_point)


class Transform:
    """
    Transformation of object

    TODO: figure out if this is really used
    """

    def __init__(self):
        self.position: Point3D
        self.rotation
        self.scale


class RegistryBase(type):
    """Metaclass for registry of scene elements"""

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)


class ArchElement(metaclass=RegistryBase):
    """Base class for architecture elements"""

    def __init__(self, id: str, type: str):
        self.id = id
        self.type = type

    @classmethod
    def from_json(cls, obj: JSONDict) -> Self:
        """Get object from JSON based on obj["type"]"""
        cls_type = obj["type"].capitalize()
        return cls.REGISTRY[cls_type].from_json(cls_type, obj)


class Wall(ArchElement):
    """Wall class in scene"""

    def __init__(self, id: str, points: list[Point3D], height: float, depth: float, sides: list, room_id: str):
        super().__init__(id, "Wall")
        self.points: list[Point3D] = points
        self.height: float = height
        self.depth: float = depth  # thickness of wall
        self.sides: list = sides  # material for each wall side
        self.room_id: str = room_id
        self.openings: list[Opening] = []

    @property
    def width(self) -> float:
        """Get width of wall"""
        d0 = self.points[0][0] - self.points[1][0]
        d1 = self.points[0][1] - self.points[1][1]
        d2 = self.points[0][2] - self.points[1][2]
        return math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

    @property
    def up(self) -> Point3D:
        """Get upward direction of wall"""
        return Point3D(0, 0, 1)  # assume z up

    @property
    def front(self) -> Point3D:
        """Get unit vector normal to front of wall"""
        dir = np.cross(self.dir.tolist(), self.up.tolist())
        return Point3D(*dir.tolist())

    @property
    def dir(self) -> Point3D:
        """Get unit vector in direction of wall"""
        dir = Point3D.sub(self.points[1], self.points[0]).normalize()
        return dir

    @property
    def centroid(self) -> Point3D:
        """Get centroid of wall"""
        mid = Point3D.mean(self.points)
        mid.z += self.height / 2
        return mid

    @property
    def front_center(self) -> Point3D:
        """Get vector to center of front face of wall"""
        mid = self.centroid
        front = self.front
        mid = Point3D.sum([mid, front.scale(self.depth / 2)])
        return mid

    @property
    def back_center(self) -> Point3D:
        """Get vector to center of back face of wall"""
        mid = self.centroid
        front = self.front
        mid = Point3D.sum([mid, front.scale(-self.depth / 2)])
        return mid

    def sample_face(self, margin: Point2D = None, is_back: bool = False) -> Point3D:
        """Sample point on face of wall"""
        sp = self.points[0]
        point = Point3D(sp[0], sp[1], sp[2])
        mw = margin[0] if margin else 0
        mh = margin[1] if margin else 0
        height = random.uniform(mh, self.height - mh)
        width = random.uniform(mw, self.width - mw)

        point = Point3D.sum([point, self.dir.scale(width)])
        point = Point3D.sum([point, self.up.scale(height)])
        sign = -1 if is_back else +1
        point = Point3D.sum([point, self.front.scale(sign * self.depth / 2)])
        return point

    @property
    def bbox3D(self) -> BBox3D:
        """Get axis-aligned bounding box around wall"""
        # assume z-up
        bbox = BBox3D.from_point_list(self.points)
        dims = bbox.dims
        if dims.y > dims.x:
            bbox.min.x -= self.depth / 2
            bbox.max.x += self.depth / 2
        else:
            bbox.min.y -= self.depth / 2
            bbox.max.y += self.depth / 2
        bbox.max.z += self.height
        return bbox

    def translate(self, vec: np.array) -> Self:
        self.points = [pts.translate(vec) for pts in self.points]
        return self

    def to_json(self) -> JSONDict:
        """Get JSON form of Wall object"""
        holes = []
        for opening in self.openings:
            if isinstance(opening, Window):
                hole_type = "Window"
            elif isinstance(opening, Door):
                hole_type = "Door"
            else:
                continue
            hole = {
                "id": opening.id,
                "type": hole_type,
                "box": {
                    "min": [opening.mid * self.width - opening.width / 2, opening.elevation - opening.height / 2],
                    "max": [opening.mid * self.width + opening.width / 2, opening.elevation + opening.height / 2],
                },
            }
            holes.append(hole)
        return {
            "id": self.id,
            "type": self.type,
            "roomId": self.room_id,
            "points": [pt.tolist() for pt in self.points],
            "height": self.height,
            "depth": self.depth,
            "materials": self.sides,
            "holes": holes,
        }

    @classmethod
    def from_json(cls, obj) -> Self:
        """Get Wall object from JSON specification"""
        assert obj["type"] == "Wall"
        wall = Wall(
            obj["id"],
            [Point3D(*p) for p in obj["points"]],
            obj["height"],
            obj.get("depth"),
            obj.get("materials"),
            obj["roomId"],
        )
        return wall


class ArchHorizontalPlane(ArchElement):
    """Base class for architecture elements which are flat in nature (e.g. floor, ceiling)"""

    def __init__(self, id: str, type: str, points: list[Point3D], depth: float, sides: list, room_id: str):
        super().__init__(id, type)
        self.points: list[Point3D] = points
        self.depth: float = depth
        self.sides: list = sides  # material
        self.room_id: str = room_id

    @property
    def bbox3D(self) -> BBox3D:
        """Get axis-aligned bounding box around architectural element"""
        # assume z-up
        bbox = BBox3D.from_point_list(self.points)
        if self.depth is not None:
            bbox.max.z += self.depth
        else:
            bbox.max.z += 0.05  # default depth TODO: push to some constants
        return bbox

    def translate(self, vec: np.array) -> Self:
        self.points = [pts.translate(vec) for pts in self.points]
        return self

    def to_json(self) -> JSONDict:
        """Convert object to JSON format"""
        return {
            "id": self.id,
            "type": self.type,
            "roomId": self.room_id,
            "points": [pt.tolist() for pt in self.points],
            "depth": self.depth,
            "materials": self.sides,
        }


class Floor(ArchHorizontalPlane):
    """Floor object in scene"""

    def __init__(self, id, points, height, sides, room_id):
        super().__init__(id, "Floor", points, height, sides, room_id)

    @classmethod
    def from_json(cls, obj) -> Self:
        """Get Floor object from JSON"""
        assert obj["type"] == "Floor"
        element = Floor(
            obj["id"], [Point3D(*p) for p in obj["points"]], obj.get("depth"), obj.get("materials"), obj.get("roomId")
        )
        return element

    def contains(self, bbox: BBox3D, wall_depth: float = 0.0) -> bool:
        floor = Polygon([(p.x, p.y) for p in self.points])
        floor_with_walls = floor.buffer(-wall_depth / 2, cap_style="square")
        box_proj = Polygon(
            [(bbox.min.x, bbox.min.y), (bbox.min.x, bbox.max.y), (bbox.max.x, bbox.max.y), (bbox.max.x, bbox.min.y)]
        )
        return box_proj.within(floor_with_walls)


class Ceiling(ArchHorizontalPlane):
    """Ceiling object in scene"""

    def __init__(self, id, points, height, sides, room_id):
        super().__init__(id, "Ceiling", points, height, sides, room_id)

    @classmethod
    def from_json(cls, obj) -> Self:
        """Get Ceiling object from JSON"""
        assert obj["type"] == "Ceiling"
        element = Ceiling(
            obj["id"], [Point3D(*p) for p in obj["points"]], obj.get("depth"), obj.get("materials"), obj.get("roomId")
        )
        return element


@dataclass
class Opening(ArchElement):
    """Base class for openings in architecture elements"""

    parent: ArchElement  # arch element associated with this opening

    def to_json(self) -> JSONDict:
        """Convert Opening object to JSON format"""
        return {
            "id": self.id,
            "type": self.__class__.__name__,
            "parent": self.parent.id,
        }


@dataclass
class WallOpening(Opening):
    """Base class for openings in walls"""

    id: str
    mid: float  # midpoint of opening (0 to 1) relative to wall width
    width: float  # width of wall opening
    height: float
    elevation: float  # height of midpoint of window

    def bbox2D(self) -> BBox2D:  # returns corners of wall opening
        raise NotImplementedError


@dataclass
class Window(WallOpening):
    """Class for windows in walls"""

    type: str = "Window"

    def to_json(self) -> JSONDict:
        """Convert Window object to JSON format"""
        json_dict = super().to_json()
        json_dict.update(
            {
                "mid": self.mid,
                "width": self.width,
                "height": self.height,
                "elevation": self.elevation,
            }
        )
        return json_dict


@dataclass
class Door(WallOpening):
    """Class for doors in walls"""

    type: str = "Door"

    def to_json(self) -> JSONDict:
        """Convert Door object to JSON format"""
        json_dict = super().to_json()
        json_dict.update(
            {
                "mid": self.mid,
                "width": self.width,
                "height": self.height,
                "elevation": self.elevation,
            }
        )
        return json_dict


class Room(ArchElement):
    """Class for defining a room, including its walls, floor, and ceiling"""

    def __init__(self, id):
        super().__init__(id=id, type="Room")
        self.wall_sides = []
        self.floor = None
        self.ceiling = None

    def to_json(self) -> JSONDict:
        """Convert object to JSON format"""
        return {
            "id": self.id,
            "type": "Other",  # not sure why existing STKs are like this
            "walls": self.wall_sides,
        }

    @classmethod
    def from_json(cls, obj) -> Self:
        """Get Room object from JSON"""
        room = Room(id=obj["id"])
        room.wall_sides = obj.get("wallIds", [])
        return room

    def __repr__(self) -> str:
        return f"Room(id={self.id}, wall_sides=[{', '.join(map(str, self.wall_sides))}], floor={self.floor}, ceiling={self.ceiling})"


class ObjectInstance:
    """Class for model and associated transform in scene"""

    def __init__(self):
        self.id: str
        self.model_id: str
        self.transform: Transform


class Modification:
    """
    Specification for a modification to an object

    TODO: may need to fully specify this
    """

    def __init__(self):
        self.type
        self.object


class SceneState:
    """Specification of scene state"""

    def __init__(self):
        from libsg.arch import Architecture  # FIXME: needed to avoid circular import :(

        self.format: str
        self.arch: Architecture  # some architecture (actually inside scene)
        self.objects: list[ObjectInstance]  # objects in this scene with their transforms (actually inside scene)
        self.selected: list[str]  # ids of elements/objects most recently changed
        self.modifications: list[Modification]


class SceneContext:
    def __init__(self):
        self.state: SceneState  # current scene state
        self.position: Point3D  # "origin" of context
        self.mask: BBox3D  # box that defines mask of context


class SceneType(Enum):
    """Enum for scene specification types"""

    id = "id"
    category = "category"
    text = "text"


@dataclass
class SceneSpec:
    """Specification of scene description"""

    type: SceneType
    input: str
    format: str  # HAB, STK
    raw: Optional[str] = None
    scene_graph: Optional[JSONDict] = None
    room_type: Optional[str] = None
    arch_spec: Optional["ArchSpec"] = None


@dataclass
class ArchSpec:
    """Specification of scene description"""

    type: SceneType
    input: Optional[str]
    format: str  # HAB, STK
    room_ids: Optional[list[str]] = None
    prompt: Optional[str] = None


@dataclass
class SceneLayoutSpec:
    """Specification of scene layout"""

    # from libsg.arch import Architecture  # FIXME: needed to avoid circular import :(

    type: SceneType
    input: str
    arch: Any = None
    raw: Optional[str] = None
    graph: Optional[JSONDict] = None


@dataclass
class ObjectTemplateInstance:
    """Specification of coarse object instance"""

    label: str  # str name of object

    # TODO: use BBox3D + Transform instead of separate tensors for this
    dimensions: torch.Tensor  # tensor of dimensions for each object
    position: torch.Tensor  # tensor of positions for each object
    orientation: torch.Tensor  # tensor of angles to apply to each object
    id: Optional[str] = None  # str id of object
    description: Optional[str] = None  # str description of object
    embedding: Optional[torch.Tensor] = None


@dataclass
class SceneLayout:
    """Scene layout definition"""

    # from libsg.arch import Architecture  # FIXME: needed to avoid circular import :(

    objects: list[ObjectTemplateInstance]
    arch: Any = None
    room_type: Optional[str] = None
    shift_by_scene_centroid: bool = True

    NEAR_FLOOR_HEIGHT = 0.05
    BOX_MESH_VERTICES = [
        (1, 2, 3),
        (2, 3, 4),
        (1, 2, 5),
        (2, 5, 6),
        (8, 5, 6),
        (8, 5, 7),
        (3, 4, 8),
        (3, 7, 8),
        (1, 3, 5),
        (3, 5, 7),
        (2, 4, 6),
        (4, 6, 8),
    ]

    def export_coarse_layout(self, output: str):
        """Export coarse layout to obj file.

        The coarse layout visualization will include the 3D oriented boxes for each object in the layout.

        :param output: path to output file (obj)
        """
        from libsg.geo import Transform as GeoTransform  # TODO: fix in-line import

        # Transform boxes to their locations pre object retrieval/generation
        all_boxes = []
        for obj in self.objects:
            synset = obj.label
            if synset is not None and ("pendant_lamp" in synset or "ceiling_lamp" in synset):
                continue  # skip ceiling lamps  # TODO: ceiling lamp placement logic
            origin = np.array(obj.dimensions) / 2
            origin[2] = 0.0  # put center at bottom of object for now
            box_verts = list(product(*[[0, obj.dimensions[i]] for i in (0, 2, 1)]))  # flip y-axis to vertical
            box_verts = np.array(box_verts) - origin  # 8 x 3

            # rotate by angle
            box_verts_rotated = GeoTransform.get_rotated_vector([0, 0, obj.orientation], box_verts)

            # shift to object location
            obj_position = np.array(obj.position)
            obj_position[2] = self.NEAR_FLOOR_HEIGHT
            box_verts_final = box_verts_rotated + obj_position
            all_boxes.append(box_verts_final)

        with open(output, "w") as f:
            # generate vertices
            for bbox in all_boxes:
                for c_idx in range(bbox.shape[0]):
                    f.write(f"v {bbox[c_idx, 0]} {bbox[c_idx, 1]} {bbox[c_idx, 2]} 1.0\n")

            # generate faces
            for b_idx, bbox in enumerate(all_boxes):
                for v1, v2, v3 in self.BOX_MESH_VERTICES:
                    f.write(f"f {b_idx * 8 + v1} {b_idx * 8 + v2} {b_idx * 8 + v3}\n")

    def print_layout(self):
        """Print layout to console"""
        out = "LAYOUT:\n"
        for object_template in self.objects:
            out += (
                f" * {object_template.label}, at position {object_template.position} with orientation "
                f"{object_template.orientation} and dimensions {object_template.dimensions}\n"
            )
        logging.debug(out)


class SceneModifySpec:
    """Specification for scene modifications"""

    def __init__(self):
        self.type
        self.input


@dataclass
class ObjectSpec:
    """Specification for object to add"""

    type: str  # type is id, category, text, or embedding
    description: Optional[str] = None
    wnsynsetkey: Optional[str] = None
    dimensions: Optional[list[float]] = None
    embedding: Optional[np.ndarray] = None
    source: Optional[str] = None

    @classmethod
    def is_arch(cls, object_spec) -> bool:
        """Return True if object is a wall, ceiling, or floor"""
        if object_spec.type == "category":
            element_type = object_spec.description.lower()
            return element_type in ["wall", "ceiling", "floor"]
        else:
            return False


@dataclass
class PlacementSpec:
    """Specification of placement for an object"""

    type: str  # "placement_point", "placement_relation"
    position: Optional[Point3D] = None
    orientation: Optional[float] = None
    allow_collisions: bool = True
    reference: Optional[ObjectSpec] = None

    @classmethod
    def get_placement_reference_object(cls, placement_spec):
        if placement_spec.type == "placement_relation":
            return placement_spec.reference


class MoveSpec:
    """Specification for moving an object"""

    def __init__(self):
        self.type  # type is id, category, or text description
        self.object


class RemoveSpec:
    """Specification for removing an object"""

    def __init__(self):
        self.type  # type is id, category, or text description
        self.object
        self.remove_children  # Whether to remove support children
        self.adjust_scene  # whether to adjust other objects in scene


class InvalidSceneGraphError(Exception):
    pass


@dataclass
class Object:
    id: int
    name: str
    attributes: list[str]
    relationships: list["Relationship"] = field(default_factory=list)
    feature: Optional[np.ndarray] = None

    def __eq__(self, __other: object) -> bool:
        return (
            isinstance(__other, Object)
            and self.id == __other.id
            and self.name == __other.name
            and set(self.attributes) == set(__other.attributes)
        )


@dataclass
class Relationship:
    id: int
    type: str
    subject: Object
    target: Object
    embedding: Optional[np.ndarray] = None

    def get_end(self, obj: Object) -> Object:
        if obj != self.subject and obj != self.target:
            raise ValueError(f"Object {obj} is not part of the relationship {self}")
        return self.target if self.subject == obj else self.subject


@dataclass
class SceneGraph:
    id: Optional[str]
    objects: list[Object]
    relationships: list[Relationship]

    def validate(
        self, allowed_objects: Optional[list[str]] = None, allowed_relationships: Optional[list[str]] = None
    ) -> bool:
        # check object names
        for obj in self.objects:
            if allowed_objects is not None and obj.name not in allowed_objects:
                raise InvalidSceneGraphError(f"Found invalid object name: {obj.name}")
            if not isinstance(obj.id, int):
                raise InvalidSceneGraphError(f"Found object_id which is not an integer: {obj.id}")

        object_ids = set([obj.id for obj in self.objects])

        # check relationships
        for rel in self.relationships:
            if allowed_relationships is not None and rel.type not in allowed_relationships:
                raise InvalidSceneGraphError(f"Found invalid relationship type: {rel.type}")
            if rel.subject.id not in object_ids:
                raise InvalidSceneGraphError(f"Found subject_id which does not correspond to a real object: {rel}")
            if rel.target.id not in object_ids:
                raise InvalidSceneGraphError(f"Found target_id which does not correspond to a real object: {rel}")

        return True

    @classmethod
    def from_json(cls, data: dict, id: Optional[str] = None) -> Self:
        objects = {obj["id"]: Object(**obj) for obj in data["objects"]}
        relationships = []
        for rel in data["relationships"]:
            try:
                subject = objects[rel["subject_id"]]
            except KeyError:
                raise InvalidSceneGraphError(
                    f"Found subject_id which does not correspond to a real object: {rel}, object_ids={objects.keys()}"
                )
            try:
                target = objects[rel["target_id"]]
            except KeyError:
                raise InvalidSceneGraphError(
                    f"Found target_id which does not correspond to a real object: {rel}, object_ids={objects.keys()}"
                )
            relationships.append(
                Relationship(
                    id=rel.get("id"),
                    type=rel["type"],
                    subject=subject,
                    target=target,
                )
            )
            subject.relationships.append(relationships[-1])
            target.relationships.append(relationships[-1])
        return cls(
            id=id,
            objects=list(objects.values()),
            relationships=relationships,
        )
    
    def to_json(self, fields: Optional[list[str]] = None) -> dict:
        """
        Convert SceneGraph object to JSON-serializable dictionary.
        
        :param fields: List of fields to include in the output. If None, all fields are included.
        :return: JSON-serializable dictionary of the SceneGraph or specified fields.
        """
        all_fields = {
            "id": self.id,
            "objects": [
                {
                    "id": obj.id,
                    "name": obj.name,
                    "attributes": obj.attributes,
                }
                for obj in self.objects
            ],
            "relationships": [
                {
                    "id": rel.id,
                    "type": rel.type,
                    "subject_id": rel.subject.id,
                    "target_id": rel.target.id,
                }
                for rel in self.relationships
            ],
        }
        
        if fields is None:
            return all_fields
        else:
            return {field: all_fields[field] for field in fields if field in all_fields}
