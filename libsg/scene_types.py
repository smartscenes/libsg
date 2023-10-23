from typing import Dict
import inspect
import math
import numpy as np
import random

JSONDict = Dict[str, any]

class Point:
    SIZE = 0

    def length_sq(self_or_cls, p=None):
        if inspect.isclass(self_or_cls):
            cls = self_or_cls
        else:
            cls = type(self_or_cls)
            p = self_or_cls
        return sum([p[i]*p[i] for i in range(0,cls.SIZE)])

    def length(self_or_cls, p=None):
        return math.sqrt(self_or_cls.length_sq(p))

    @classmethod
    def min(cls, points):
        return cls(*[ min([p[i] for p in points]) for i in range(0,cls.SIZE)])

    @classmethod
    def max(cls, points):
        return cls(*[ max([p[i] for p in points]) for i in range(0,cls.SIZE)])

    @classmethod
    def sum(cls, points):
        return cls(*[ sum([p[i] for p in points]) for i in range(0,cls.SIZE)])

    @classmethod
    def mean(cls, points):
        k = len(points)
        return cls(*[ sum([p[i] for p in points])/k for i in range(0,cls.SIZE)])

    @classmethod
    def sub(cls, a, b):
        return cls(*[ (a[i] - b[i]) for i in range(0,cls.SIZE)])

    @classmethod
    def add(cls, a, b):
        return cls(*[ (a[i] + b[i]) for i in range(0,cls.SIZE)])

    @classmethod
    def distance_sq(cls, a, b):
        return sum([ (a[i] - b[i])*(a[i] - b[i]) for i in range(0,cls.SIZE)])

    @classmethod
    def distance(cls, a, b):
        return math.sqrt(cls.distance_sq(a, b))

    @classmethod
    def fromlist(cls, l):
        return cls(*[p for p in l])


class Point2D(Point):
    SIZE = 2
    def __init__(self, x, y):
        super().__init__()
        self.x: float = x
        self.y: float = y

    def __str__(self):
        return str(self.tolist())

    def __getitem__(self, key):
        return (self.x, self.y)[key]

    def tolist(self):
        return [self.x,self.y]


class Point3D(Point):
    SIZE = 3
    def __init__(self, x, y, z):
        super().__init__()
        self.x: float = x
        self.y: float = y 
        self.z: float = z

    def __str__(self):
        return str(self.tolist())

    def __getitem__(self, key):
        return (self.x, self.y, self.z)[key]

    def tolist(self):
        return [self.x,self.y,self.z]

    def scale(self, s):
        self.x = self.x * s
        self.y = self.y * s
        self.z = self.z * s
        return self

    def normalize(self):
        length = self.length()
        self.scale(1/length)
        return self


class BBox2D:
    def __init__(self):
        self.min: Point2D
        self.max: Point2D

    @classmethod
    def from_point_list(cls, points):
        min_point = Point2D.min(points)
        max_point = Point2D.max(points)
        return BBox2D(min_point, max_point)

class BBox3D:

    LEFT = 0
    RIGHT = 1
    BOTTOM = 2
    TOP = 3
    FRONT = 4
    BACK = 5

    def __init__(self, min, max):
        self.min: Point3D = min
        self.max: Point3D = max

    @property
    def centroid(self):
        mid = Point3D.mean([self.min, self.max])
        return mid

    @property
    def dims(self):
        point = Point3D.sub(self.max, self.min)
        return point

    def __str__(self):
        return f'BBox3D({self.min}, {self.max})'

    def get_face_center(self, index):
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

    def get_face_outnormal(self, index):
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

    def get_face_dims(self, index):
        dims = self.dims
        if index == BBox3D.LEFT or index == BBox3D.RIGHT:
            return Point3D(dims.y, dims.z, dims.x)
        elif index == BBox3D.BOTTOM or index == BBox3D.TOP:
            return Point3D(dims.x, dims.y, dims.z)
        else: 
            return Point3D(dims.x, dims.z, dims.y)

    def get_point_on_face(self, index, r, margin):
        face_dims = self.get_face_dims(index)
        p0 = self.min
        p1 = self.max
        m0 = margin[0] if margin else 0
        m1 = margin[1] if margin else 0
        d0 = r[0]*(face_dims[0]-m0) + m0
        d1 = r[1]*(face_dims[1]-m1) + m1

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

    def sample_face(self, index, margin):
        return self.get_point_on_face(index, [random.uniform(0,1), random.uniform(0,1)], margin)

    def contains(self, p):
        return (self.max.x >= p.x >= self.min.x) and (self.max.y >= p.y >= self.min.y) and (self.max.z >= p.z >= self.min.z)

    @classmethod
    def from_point_list(cls, points):
        min_point = Point3D.min(points)
        max_point = Point3D.max(points)
        return BBox3D(min_point, max_point)

    @classmethod
    def from_min_max(cls, min, max):
        min_point = Point3D.fromlist(min)
        max_point = Point3D.fromlist(max)
        return BBox3D(min_point, max_point)


class Transform:
    def __init__(self):
        self.position: Point3D
        self.rotation
        self.scale

class RegistryBase(type):
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

    def __init__(self, id: str, type: str):
        self.id = id
        self.type = type

    @classmethod
    def from_json(cls, obj):
        cls_type = obj['type'].capitalize()
        return cls.REGISTRY[cls_type].from_json(cls_type, obj)


class Wall(ArchElement):
    def __init__(self, id, points, height, depth, sides, room_id):
        super().__init__(id, 'Wall')
        self.points: list[Point3D] = points
        self.height: float = height
        self.depth: float = depth
        self.sides: list = sides  # material for each wall side
        self.room_id: str = room_id

    @property
    def width(self):
        d0 = self.points[0][0] - self.points[1][0] 
        d1 = self.points[0][1] - self.points[1][1]
        return math.sqrt(d0*d0 + d1*d1)

    @property
    def up(self):
        return Point3D(0,0,1) # assume z up

    @property
    def front(self):
        dir = np.cross(self.dir.tolist(), self.up.tolist())
        return Point3D(*dir.tolist())

    @property
    def dir(self):
        dir = Point3D.sub(self.points[1], self.points[0]).normalize()
        return dir

    @property
    def centroid(self):
        mid = Point3D.mean(self.points)
        mid.z += self.height / 2
        return mid

    @property
    def front_center(self):
        mid = self.centroid
        front = self.front
        mid = Point3D.sum([mid, front.scale(self.depth / 2)])
        return mid

    @property
    def back_center(self):
        mid = self.centroid
        front = self.front
        mid = Point3D.sum([mid, front.scale(-self.depth / 2)])
        return mid

    def sample_face(self, margin: Point2D=None, is_back=False):
        sp = self.points[0]
        point = Point3D(sp[0], sp[1], sp[2])
        mw = margin[0] if margin else 0
        mh = margin[1] if margin else 0
        height = random.uniform(mh, self.height-mh)
        width = random.uniform(mw, self.width-mw)

        point = Point3D.sum([point, self.dir.scale(width)])
        point = Point3D.sum([point, self.up.scale(height)])
        sign = -1 if is_back else +1
        point = Point3D.sum([point, self.front.scale(sign*self.depth / 2)])
        return point

    @property
    def bbox3D(self):
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

    def to_json(self):
        return {
            "id": self.id,
            "type": self.type,
            "roomId": self.room_id,
            "points": self.points,
            "height": self.height,
            "depth": self.depth,
            "materials": self.sides
        }

    def from_json(cls, obj):
        assert(obj['type'] == 'Wall')
        wall = Wall(obj['id'], obj['points'], obj['height'], obj.get('depth'), obj.get('materials'), obj.get('roomId'))
        return wall

class ArchHorizontalPlane(ArchElement):
    def __init__(self, id, type, points, depth, sides, room_id):
        super().__init__(id, type)
        self.points: list[Point3D] = points
        self.depth: float = depth
        self.sides: list = sides  # material for each wall side
        self.room_id: str = room_id

    @property
    def bbox3D(self):
        # assume z-up
        bbox = BBox3D.from_point_list(self.points)
        if self.depth is not None:
            bbox.max.z += self.depth
        else: 
            bbox.max.z += 0.05 # default depth TODO: push to some constants
        return bbox

    def to_json(self):
        return {
            "id": self.id,
            "type": self.type,
            "roomId": self.room_id,
            "points": self.points,
            "depth": self.depth,
            "materials": self.sides
        }

class Floor(ArchHorizontalPlane):
    def __init__(self, id, points, height, sides, room_id):
        super().__init__(id, 'Floor', points, height, sides, room_id)

    def from_json(cls, obj):
        assert(obj['type'] == 'Floor')
        element = Floor(obj['id'], obj['points'], obj.get('depth'), obj.get('materials'), obj.get('roomId'))
        return element

class Ceiling(ArchHorizontalPlane):
    def __init__(self, id, points, height, sides, room_id):
        super().__init__(id, 'Ceiling', points, height, sides, room_id)

    def from_json(cls, obj):
        assert(obj['type'] == 'Ceiling')
        element = Ceiling(obj['id'], obj['points'], obj.get('depth'), obj.get('materials'), obj.get('roomId'))
        return element


class Opening(ArchElement):
    def __init__(self):
        self.parent: ArchElement  # arch element associated with this opening

class WallOpening(Opening):
    def __init__(self):
        self.mid: float  # midpoint of opening (0 to 1) relative to wall width
        self.width: float  # width of wall opening
        self.height: float
        self.elevation: float

    def bbox2D(self) -> BBox2D:  # returns corners of wall opening
        pass

class Window(WallOpening):
    pass

class Door(WallOpening):
    pass

class Room(ArchElement):
    def __init__(self):
        self.wall_sides
        self.floor
        self.ceiling

class Arch:
    def __init__(self):
        self.elements: list(ArchElement)  # flat list of architecture elements (floor, ceiling, wall, and other stuff)
        self.rooms: list(Room)  # list of rooms
        self.openings: list(Opening)  # flat list opening (windows, doors, and relevant information)

class ObjectInstance:
    def __init__(self):
        self.id: str
        self.model_id: str
        self.transform: Transform

class Modification:
    def __init__(self):
        self.type
        self.object

class SceneState:
    def __init__(self):
        self.format: str        
        self.arch: Arch  # some architecture (actually inside scene)
        self.objects: list[ObjectInstance]  # objects in this scene with their transforms (actually inside scene)
        self.selected: list[str]  # ids of elements/objects most recently changed
        self.modifications: list[Modification]


class SceneContext:
    def __init__(self):
        self.state: SceneState  # current scene state
        self.position: Point3D  # "origin" of context
        self.mask: BBox3D  # box that defines mask of context

class SceneSpec:
    def __init__(self, type, input, format):
        self.type = type  # type is id, category, or text description
        self.input = input
        self.format = format

class SceneModifySpec:
    def __init__(self):
        self.type
        self.input

class PlacementSpec:
    def __init__(self):
        self.type  # type is point3d or relationship wrt another object
        self.placement

    @classmethod
    def get_placement_reference_object(cls, placement_spec):
        if placement_spec.type == 'placement_relation': 
            return placement_spec.reference

class ObjectSpec:
    def __init__(self):
        self.type  # type is id, category, or text description
        self.object

    @classmethod
    def is_arch(cls, object_spec):
        if object_spec.type == 'category': 
            element_type = object_spec.object.lower()
            return element_type in ['wall','ceiling','floor']
        else:
            return False

class MoveSpec:
    def __init__(self):
        self.type  # type is id, category, or text description
        self.object

class RemoveSpec:
    def __init__(self):
        self.type  # type is id, category, or text description
        self.object 
        self.remove_children # Whether to remove support children
        self.adjust_scene  # whether to adjust other objects in scene
