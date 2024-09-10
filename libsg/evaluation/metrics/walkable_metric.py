import numpy as np
import trimesh, os, cv2

from .base import EvaluationBase
from libsg.scene import Scene
from libsg.scene_types import SceneGraph

# https://github.com/PhyScene/PhyScene/blob/main/scripts/eval/walkable_metric.py
def cal_walkable_metric(floor_plan, floor_plan_centroid, bboxes, doors, robot_width=0.01, visual_path=None, calc_object_area=False):

    vertices, faces = floor_plan
    vertices = vertices - floor_plan_centroid
    vertices = vertices[:, 0::2]
    scale = np.abs(vertices).max()+0.2
    bboxes = bboxes[bboxes[:, 1] < 1.5]

    # door
    if doors is not None:
        doors_position = [door[0][door[0][:, 1] == door[0][:, 1].min()].mean(0) for door in doors]

    image_size = 256
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    robot_width = int(robot_width / scale * image_size/2)

    def map_to_image_coordinate(point):
        x, y = point
        x_image = int(x / scale * image_size/2)+image_size/2
        y_image = int(y / scale * image_size/2)+image_size/2
        return x_image, y_image

    # draw face
    for face in faces:
        face_vertices = vertices[face]
        face_vertices_image = [
            map_to_image_coordinate(v) for v in face_vertices]

        pts = np.array(face_vertices_image, np.int32)
        pts = pts.reshape(-1, 1, 2)
        color = (255, 0, 0)  # Blue (BGR)
        cv2.fillPoly(image, [pts], color)

    kernel = np.ones((robot_width, robot_width))
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
    # draw bboxes
    # cv2.imwrite("image.png", image)
    for box in bboxes:
        center = map_to_image_coordinate(box[:3][0::2])
        size = (int(box[3] / scale * image_size / 2) * 2,
                int(box[5] / scale * image_size / 2) * 2)
        angle = box[-1]

        # calculate box vertices
        box_points = cv2.boxPoints(
            ((center[0], center[1]), size, -angle/np.pi*180))
        box_points = np.intp(box_points)

        cv2.drawContours(image, [box_points], 0,
                         (0, 255, 0), robot_width)  # Green (BGR)
        cv2.fillPoly(image, [box_points], (0, 255, 0))

    cv2.imwrite("image.png", image)

    if calc_object_area:
        green_cnt = 0
        blue_cnt = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if list(image[i][j]) == [0, 255, 0]:
                    green_cnt += 1
                elif list(image[i][j]) == [255, 0, 0]:
                    blue_cnt += 1
        object_area_ratio = green_cnt/(blue_cnt+green_cnt)
        
    walkable_map = image[:, :, 0].copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        walkable_map, connectivity=8)

    if doors is not None:
        walkable_rate_list = []
        for door_position in doors_position:
            distance_to_door = np.inf
            door_position = door_position - floor_plan_centroid
            door_position = map_to_image_coordinate(
                np.array(door_position)[0::2])
            door_position = np.array(door_position, np.int32)
            cv2.circle(image, door_position, 5, (255, 255, 255), 5)
            if visual_path is not None:
                cv2.imwrite(visual_path, image)
            
            walkable_map_max = np.zeros_like(walkable_map)
            walkable_map_door = np.zeros_like(walkable_map)
            for label in range(1, num_labels):
                mask = np.zeros_like(walkable_map)
                mask[labels == label] = 255

                dist_transform = cv2.distanceTransform(
                    (255-mask).T, distanceType=cv2.DIST_L2, maskSize=5)
                distance_mask_to_door = dist_transform[door_position[0], door_position[1]]
                
                if distance_mask_to_door < distance_to_door and distance_mask_to_door <= robot_width + 1:
                    # room connected component with door
                    distance_to_door = distance_mask_to_door
                    walkable_map_door = mask.copy()
            walkable_rate = walkable_map_door.sum()/walkable_map.sum()
            walkable_rate_list.append(walkable_rate)
        # print("walkable_rate:", np.mean(walkable_rate_list))
        if calc_object_area:
            return np.mean(walkable_rate_list),object_area_ratio
        else:
            return np.mean(walkable_rate_list),None
    else:
        walkable_map_max = np.zeros_like(walkable_map)
        for label in range(1, num_labels):
            mask = np.zeros_like(walkable_map)
            walkable_map_door = np.zeros_like(walkable_map)
            mask[labels == label] = 255

            if mask.sum() > walkable_map_max.sum():
                # room connected component with door
                walkable_map_max = mask.copy()

            # print("walkable_rate:", walkable_map_max.sum()/walkable_map.sum())
            if calc_object_area:
                return walkable_map_max.sum()/walkable_map.sum(), object_area_ratio
            else:
                return walkable_map_max.sum()/walkable_map.sum()
        if calc_object_area:    
            return 0.,object_area_ratio
        else:
            return 0.,None

class WalkableMetric(EvaluationBase):
    def __init__(self, object_dir_mapping: str, robot_width=0.3, num_iterations=100, **kwargs):
        super().__init__()
        self.robot_width = robot_width
        self.num_iterations = num_iterations
        self.object_dir_mapping = object_dir_mapping
        self.reset()
        
    def reset(self):
        self.walkable_rates = []
        self.object_area_ratios = []
        self.num_scenes = 0

    def _load_scene(self, scene: Scene):
        bboxes = np.empty((0, 7))

        for mi in scene.model_instances:
            
            # TODO: change this to a solr query?
            # TODO: make this more general to support other asset sources
            asset_source, _, model_id = mi.model_id.partition(".")
            object_path = os.path.join(
                self.object_dir_mapping[asset_source], str(model_id[0]), f"{model_id}.collider.glb"
            )

            if(not os.path.exists(object_path)):
                continue
            mesh = trimesh.load(object_path, force="mesh")
            
            # Get the center coordinates of the bounding box
            bounds = mesh.bounds
            center = (bounds[0] + bounds[1]) / 2

            # Get the width, height, and depth of the bounding box
            extents = mesh.extents
            width = extents[0]
            height = extents[1]
            depth = extents[2]

            # Get the rotation angle of the bounding box
            longest_edge = np.argmax(extents)
            
            # Get the orientation of the mesh
            orientation = mesh.principal_inertia_vectors[longest_edge]
            
            # Calculate the angle between the orientation and the X-axis
            angle = np.arctan2(orientation[1], orientation[0])
            # print("Angle: ", angle)
            new_row = np.array([center[0], center[1], center[2], width, height, depth, angle])
            bboxes = np.vstack((bboxes, new_row))
        return bboxes
    
    def __call__(self, inp, scene_graph: SceneGraph, scene: Scene):
        floor_points, floor_faces, room_id = scene.arch.get_floor_plan()

        floor_plan = (np.array([[p.x, p.z, p.y] for p in floor_points]), floor_faces)
        # print(floor_plan, floor_plan)
        floor_plan_centroid = np.mean(floor_plan[0], axis=0)
        
        # doors = scene.arch.get_door_positions(room_id)
        # print(doors)
        bbox = self._load_scene(scene)
        
        doors = None
        
        result, object_area_ratio = cal_walkable_metric(
            floor_plan=floor_plan,
            floor_plan_centroid=floor_plan_centroid,
            bboxes=bbox,
            doors=doors,
            robot_width=self.robot_width,
            calc_object_area=True
        )
        
        # print("Walkable Metric: ", result, "Object Area Ratio: ", object_area_ratio)
        
        self.num_scenes += 1
        self.walkable_rates.append(result)
        self.object_area_ratios.append(object_area_ratio)


    def log(self):
        result = np.mean(self.walkable_rates)
        object_area_ratio = np.mean(self.object_area_ratios)
        print(f"Walkable Metric: {result} Object Area Ratio: {object_area_ratio}")
        return result