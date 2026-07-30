import torch


class RobotSegmenter:
    def __init__(self, kinematics, distance_threshold=0.05, use_cuda_graph=True, ops_dtype=torch.bfloat16):
        self.kinematics = kinematics
        self.distance_threshold = distance_threshold

    @staticmethod
    def get_pointcloud_from_depth(self, camera_obs):
        return camera_obs.get_pointcloud(project_to_pose=True)

    def update_camera_projection(self, camera_obs):
        camera_obs.update_projection_rays()

    def get_robot_mask_from_active_js(self, camera_obs, active_joint_state):
        points = camera_obs.get_pointcloud(project_to_pose=True)
        spheres = self.kinematics.compute_kinematics(active_joint_state).robot_spheres
        flat = points.reshape(points.shape[0], -1, 3)
        centers, radii = spheres[..., :3], spheres[..., 3]
        distance = torch.cdist(flat, centers.reshape(centers.shape[0],-1,3))-radii.reshape(radii.shape[0],1,-1)
        mask = (distance.min(-1).values < self.distance_threshold).reshape(points.shape[:-1])
        filtered = torch.where(mask, torch.zeros_like(camera_obs.depth_image), camera_obs.depth_image)
        return mask, filtered

    get_robot_mask = get_robot_mask_from_active_js
