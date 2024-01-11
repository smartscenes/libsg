import os

import numpy as np
import pybullet as p
import pygltflib
from scipy.spatial.transform import Rotation

from libsg.simulator import Simulator
from libsg.geo import Transform

class SimScene:
    def __init__(self, sim: Simulator, scene, asset_paths, include_ground=False):
        self.sim = sim
        self.scene = scene
        self.asset_paths = asset_paths
        self._create_scene_geometry(include_ground)

    @property
    def use_y_up(self):
        return self.sim._use_y_up

    def _create_scene_geometry(self, include_ground=False):
        use_y_up = self.use_y_up
        for el in self.scene.arch.elements:
            if el.type == 'Wall':
                SimScene.add_wall(self.sim, el, use_y_up=use_y_up)
        for mi in self.scene.modelInstances:
            SimScene.add_object(self.sim, self.asset_paths['collision_mesh_dir'], mi, static=True)
        self.sim.set_single_collection_filter_group()
        self.aabb = self.sim.get_aabb_all()
        if include_ground:
            dims = self.aabb.dims
            d0 = dims[0]
            pos = self.aabb.centroid.tolist()
            if use_y_up:
                d1 = dims[2]
                pos[1] = 0 #self.aabb.min[1]
            else:
                d1 = dims[1]
                pos[2] = 0 #self.aabb.min[2]
            self.ground = SimScene.add_ground(self.sim, d0, d1, 0.01, pos=pos, use_y_up=use_y_up)

    @staticmethod
    def add_ground(sim, w, h, depth, pos=[0.0,0.0,0.0], use_y_up=False):
        if use_y_up:
            half_extents = np.array([w, depth, h]) * 0.5
        else:
            half_extents = np.array([w, h, depth]) * 0.5
        transform = Transform()
        transform.set_translation(pos)
        return sim.add_box(obj_id='ground', half_extents=half_extents, transform=transform, static=True)

    @staticmethod
    def add_wall(sim, node, use_y_up=False):
        h = node.height
        p0 = np.array(node.points[0])
        p1 = np.array(node.points[1])
        c = (p0 + p1) * 0.5
        if use_y_up:
            c[1] += h * 0.5
        else:
            c[2] += h * 0.5
        dp = p1 - p0
        dp_l = np.linalg.norm(dp)
        dp = dp / dp_l
        angle = np.arccos(dp[0])
        if use_y_up:
            rot_q = Rotation.from_euler('y', angle).as_quat()
            half_extents = np.array([dp_l, h, node.depth]) * 0.5
        else:
            rot_q = Rotation.from_euler('z', angle).as_quat()
            half_extents = np.array([dp_l, node.depth, h]) * 0.5
        scale = np.array([1.0, 1.0, 1.0])
        transform = Transform.from_rts(rot_q, np.squeeze(c), scale)
        return sim.add_box(obj_id=node.id, half_extents=half_extents,
                        transform=transform, static=True)

    @staticmethod
    def convert_glb_to_obj(glb_path, obj_path):
        import struct
        # Load the glb file
        glb = pygltflib.GLTF2().load(glb_path)

        # get the first mesh in the current scene (in this example there is only one scene and one mesh)
        mesh = glb.meshes[glb.scenes[glb.scene].nodes[0]]

        # get the vertices for each primitive in the mesh (in this example there is only one)
        all_triangles = []
        vertices = []
        for primitive in mesh.primitives:
            triangles_accessor = glb.accessors[primitive.indices]
            triangles_buffer_view = glb.bufferViews[triangles_accessor.bufferView]
            triangles = np.frombuffer(
                glb.binary_blob()[
                    triangles_buffer_view.byteOffset
                    + triangles_accessor.byteOffset : triangles_buffer_view.byteOffset
                    + triangles_buffer_view.byteLength
                ],
                dtype="uint8",
                count=triangles_accessor.count,
            )
            triangles = triangles.reshape((-1, 3))
            all_triangles.extend(triangles.tolist())

            # get the binary data for this mesh primitive from the buffer
            accessor = glb.accessors[primitive.attributes.POSITION]
            bufferView = glb.bufferViews[accessor.bufferView]
            buffer = glb.buffers[bufferView.buffer]
            data = glb.get_data_from_buffer_uri(buffer.uri)

            # pull each vertex from the binary buffer and convert it into a tuple of python floats
            for i in range(accessor.count):
                index = bufferView.byteOffset + accessor.byteOffset + i*12  # the location in the buffer of this vertex
                d = data[index:index+12]  # the vertex data
                v = struct.unpack("<fff", d)   # convert from base64 to three floats
                vertices.append(v)
                print(i, v)

        # convert a numpy array for some manipulation

        # Write to obj file
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        with open(obj_path, 'w') as obj_file:
            for vertex in vertices:
                obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            for triangle in triangles:
                obj_file.write(f"f {triangle[0]+1} {triangle[1]+1} {triangle[2]+1}\n")

    @staticmethod
    def add_object(sim: Simulator, data_dir, node, static):
        model_id = node.model_id.split('.')[1]
        if len(model_id) < 16:  # skip windows/doors
            return
        transform = node.transform
        col_obj_filename = os.path.join(data_dir, "obj", f"{model_id}.obj")
        if not os.path.exists(col_obj_filename):
            col_glb_filename = os.path.join(data_dir, model_id[0], f"{model_id}.collider.glb")
            SimScene.convert_glb_to_obj(col_glb_filename, col_obj_filename)
        return sim.add_mesh(obj_id=node.id, obj_file=col_obj_filename,
                            transform=transform, vis_mesh_file=None, static=static)
