import copy
from typing import Optional, Self

import numpy as np
from scipy.spatial.transform import Rotation
from libsg.scene_types import JSONDict


class Transform:
    """General class to handle arbitrary transforms of objects."""

    def __init__(self):
        self._r = [0.0, 0.0, 0.0, 1.0]  # quaternion [x,y,z,w]
        self._t = [0.0, 0.0, 0.0]  # [x,y,z]
        self._s = [1.0, 1.0, 1.0]  # [sx,sy,sz]
        self._mat4 = np.identity(4)

    @property
    def translation(self):
        return self._t

    def set_translation(self, t):
        self._t = copy.deepcopy(t) if isinstance(t, list) else copy.deepcopy(t.tolist())
        self.__update_matrix()

    @property
    def rotation(self):
        return self._r

    def set_rotation(self, r):
        self._r = copy.deepcopy(r)
        self.__update_matrix()

    @property
    def scale(self):
        return self._s

    def set_scale(self, s):
        if isinstance(s, (int, float)):
            self._s = [float(s)] * 3
        else:
            self._s = copy.deepcopy(s) if isinstance(s, list) else copy.deepcopy(s.tolist())
        self.__update_matrix()

    @property
    def mat4(self):
        return self._mat4

    def __update_matrix(self):
        self._mat4 = Transform.rts_to_mat4(self._r, self._t, self._s)

    @classmethod
    def rts_to_mat4(cls, rotation, translation, scale):
        # TODO(MS) test correctness
        T = np.identity(4)
        T[:3, 3] = translation
        R = np.identity(4)
        R[:3, :3] = Rotation.from_quat(rotation).as_matrix()
        S = np.identity(4)
        S[:3, :3] = np.diag(scale)
        X = (T @ R @ S).transpose()
        return X

    @classmethod
    def mat4_to_rts(cls, mat4):
        # TODO(MS) test correctness
        mat4 = np.asarray(mat4).reshape((4, 4)).transpose()
        translation = mat4[:3, 3]
        scale = np.linalg.norm(mat4, axis=0)[:3]
        scale_zeros = np.isclose(scale, [0, 0, 0])
        if np.any(scale_zeros):
            # TODO(MS): hack to avoid division by zero
            scale[scale_zeros] = 1e-7
        R = mat4[:3, :3] / scale
        if np.linalg.det(R) < 0:  # if reflection, flip one axis and negate scale
            R[:3, 0] *= -1
            scale[0] *= -1
        rotation = Rotation.from_matrix(R).as_quat().tolist()
        return rotation, translation, scale

    @classmethod
    def from_mat4(cls, mat4):
        xform = Transform()
        xform._mat4 = np.asarray(mat4).reshape((4, 4))
        xform._r, xform._t, xform._s = Transform.mat4_to_rts(mat4)
        return xform

    @classmethod
    def from_rts(cls, rotation, translation, scale):
        xform = Transform()
        xform._r = copy.deepcopy(rotation)
        xform._t = copy.deepcopy(translation)
        xform._s = copy.deepcopy(scale)
        xform._mat4 = Transform.rts_to_mat4(rotation, translation, scale)
        return xform

    def rightMultiply(self, o):
        # TODO(MS) check correctness
        self._mat4 = self._mat4 @ o._mat4
        self._r, self._t, self._s = Transform.mat4_to_rts(self._mat4)
        return self

    def to_json(self, obj=None) -> JSONDict:
        obj = obj if obj else {}
        obj["rows"] = 4
        obj["cols"] = 4
        obj["data"] = self._mat4.flatten().tolist()
        rotation, translation, scale = Transform.mat4_to_rts(self._mat4)
        obj["rotation"] = list(rotation)  # x,y,z,w
        obj["translation"] = list(translation)
        obj["scale"] = list(scale)
        return obj

    @classmethod
    def from_json(cls, obj):
        mat4 = obj["data"]
        xform = Transform()
        xform._mat4 = np.asarray(mat4).reshape((4, 4))
        xform._r, xform._t, xform._s = Transform.mat4_to_rts(mat4)
        return xform

    @classmethod
    def axis_pair_to_basis(cls, v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        return np.column_stack((v1, v2, v3))

    @classmethod
    def get_alignment_matrix(cls, source_up, source_front, target_up, target_front):
        """
        Returns matrix to align from objectUp/objectFront to targetUp/targetFront
        Assumptions: objectUp perpendicular to objectFront, targetUp perpendicular to targetFront.
        :param vector objectUp Object's semantic up vector
        :param vector objectFront Object's semantic front vector
        :param vector targetUp Target up vector
        :param vector targetFront Target front vector
        """
        source_matrix = cls.axis_pair_to_basis(source_up, source_front)
        target_matrix = cls.axis_pair_to_basis(target_up, target_front)
        return target_matrix @ np.linalg.inv(source_matrix)

    @classmethod
    def get_alignment_quaternion(cls, source_up, source_front, target_up, target_front):
        """
        Returns quaternion to align from objectUp/objectFront to targetUp/targetFront
        Assumptions: objectUp perpendicular to objectFront, targetUp perpendicular to targetFront.
        :param vector objectUp Object's semantic up vector
        :param vector objectFront Object's semantic front vector
        :param vector targetUp Target up vector
        :param vector targetFront Target front vector
        """
        matrix = cls.get_alignment_matrix(source_up, source_front, target_up, target_front)
        return Rotation.from_matrix(matrix).as_quat().tolist()

    @classmethod
    def get_rotated_vector(cls, rotvec, v):
        """
        Returns a rotated version of vector v around axis & angle specified by rotvec.
        Assumptions: objectUp perpendicular to objectFront, targetUp perpendicular to targetFront.
        :param vector rotvec Vector specifying axis and angle in radians by which to rotate
        :param vector v Vector to rotate
        """
        return Rotation.from_rotvec(np.asarray(rotvec)).apply(np.asarray(v)).tolist()

    def swap_axes(self, axis_1: int, axis_2: int) -> Self:
        """Return a new Transform with two axes swapped.

        TODO: not tested

        :param axis_1: index of first axis to swap
        :param axis_2: index of second axis to swap
        :return: same transform with axes swapped.
        """
        if axis_1 == axis_2:
            return self

        # construct transformation matrix
        transform_matrix = np.zeros((4, 4), dtype=float)
        transform_matrix[3, 3] = 1
        rotation_matrix = np.eye(3)
        row_1 = np.copy(rotation_matrix[axis_1])
        row_2 = np.copy(rotation_matrix[axis_2])
        rotation_matrix[axis_1, :] = row_2
        rotation_matrix[axis_2, :] = row_1
        transform_matrix[:3, :3] = rotation_matrix

        # apply transformation
        self._mat4 = transform_matrix @ self._mat4
        self._r, self._t, self._s = Transform.mat4_to_rts(self._mat4)
        return self

    def invert_axis(self, axis: int) -> Self:
        """Returns transform with specified axis inverted (negated).

        TODO: not tested

        :param axis: index of axis to invert
        :return: transform with axis inverted
        """
        transform_matrix = np.eyes((4, 4), dtype=float)
        transform_matrix[axis, :3] = -transform_matrix[axis, :3]

        self._mat4 = transform_matrix @ self._mat4
        self._r, self._t, self._s = Transform.mat4_to_rts(self._mat4)
        return self

    def rotate(self, axis: int, rotate: float) -> Self:
        """Rotates transform around specified axis, including both position and orientation.

        TODO: not tested

        :param axis: index of axis to invert
        :param rotate: angle in radians to rotate
        :return: transform with specified rotation and orientation updated
        """
        rotvec = np.zeros((3,), dtype=float)
        rotvec[axis] = rotate
        rot = Rotation.from_rotvec(rotvec)

        # apply rotation to translation and compose with rotation of object
        self.transform.set_translation((rot.as_matrix() @ np.array(self.transform.translation)).tolist())
        new_rot = rot * Rotation.from_quat(self.transform.rotation)
        self.transform.set_rotation(new_rot.as_quat().tolist())
        return self
