import numpy as np

from scipy.spatial.transform import Rotation
from libsg.geo import Transform

class SimScene:
    def __init__(self, sim, scene, asset_paths, include_ground=False):
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
    def add_object(sim, data_dir, node, static):
        model_id = node.model_id.split('.')[1]
        if len(model_id) < 16:  # skip windows/doors
            return
        transform = node.transform
        col_obj_filename = f'{data_dir}/{model_id}.obj'
        return sim.add_mesh(obj_id=node.id, obj_file=col_obj_filename,
                            transform=transform, vis_mesh_file=None, static=static)
