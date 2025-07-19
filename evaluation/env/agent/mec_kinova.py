import os
import torch
import trimesh
import numpy as np
from typing import List, Optional, Union, Tuple
from math import pi
from yourdfpy import URDF, Link
from utils.path import RootPath

class MecKinova:
    JOINTS_NAMES = [
        'base_y_base_x', 
        'base_theta_base_y', 
        'base_link_base_theta', 
        'joint_1',
        'joint_2', 
        'joint_3', 
        'joint_4', 
        'joint_5', 
        'joint_6', 
        'joint_7'
    ]

    # The upper and lower bounds need to be symmetric with the origin
    JOINT_LIMITS = np.array(
        [
            (-6.0, 6.0), # (-30, 30)
            (-6.0, 6.0), # (-30, 30)
            (-pi, pi),
            (-pi, pi),
            (-2.24, 2.24),
            (-pi, pi),
            (-2.57, 2.57),
            (-pi, pi),
            (-2.09, 2.09),
            (-pi, pi),
        ]
    )

    # The upper and lower bounds need to be symmetric with the origin
    ACTION_LIMITS = np.array(
        [
            (-0.25, 0.25),
            (-0.25, 0.25),
            (-0.25, 0.25),
            (-0.25, 0.25),
            (-0.20, 0.20),
            (-0.15, 0.15),
            (-0.20, 0.20),
            (-0.10, 0.10),
            (-0.20, 0.20),
            (-0.10, 0.10),
        ]
    )
    
    # Tuples of radius in meters and the corresponding links, values are centers on that link
    SPHERES = [
        (
            0.02,
            {
                "bracelet_link": [
                    [0.17, -0.02, 1.65],
                    [0.05, -0.02, 1.65],
                    [0.155, -0.025, 1.7],
                    [0.065, -0.025, 1.7],
                ],
            }
        ),
        (
            0.05,
            {
                "spherical_wrist_2_link": [[0.11, -0.02, 1.5]],
                "bracelet_link": [[0.11, -0.02, 1.6]],
            }
        ),
        (
            0.06,
            {
                "base_link": [
                    [0, -0.25, 0.35],
                    [-0.27, 0, 0.35],
                    [0.27, 0, 0.35],
                    [0, 0.25, 0.35],
                ],
                "shoulder_link": [
                    [0.1, 0, 0.4],
                    [0.1, 0, 0.5],
                    [0.1, 0, 0.6],
                ],
                "half_arm_1_link": [
                    [0.1, -0.03, 0.8],
                    [0.1, -0.03, 0.7],
                ],
                "half_arm_2_link": [
                    [0.1, 0.02, 1],
                    [0.1, 0.02, 0.9],
                ],
                "forearm_link": [
                    [0.1, -0.02, 1.2],
                    [0.1, -0.02, 1.1],
                ],
                "spherical_wrist_1_link": [
                    [0.1, -0.02, 1.35],
                ]
            },
        ),
        (
            0.07, 
            {
                "base_link": [
                    [0.24, 0.25, 0.1],
                    [-0.27, 0.25, 0.1],
                    [0.24, -0.25, 0.1],
                    [-0.27, -0.25, 0.1],
                    [0, -0.25, 0.1],
                    [-0.27, 0, 0.1],
                    [0.27, 0, 0.1],
                    [0, 0.25, 0.1],
                    [0.24, 0.25, 0.22],
                    [-0.27, 0.25, 0.22],
                    [0.24, -0.25, 0.22],
                    [-0.27, -0.25, 0.22],
                    [0.24, 0.25, 0.35],
                    [-0.27, 0.25, 0.35],
                    [0.24, -0.25, 0.35],
                    [-0.27, -0.25, 0.35],
                ]
            }
        ),
    ]

    BASE_LINK = [
        "virtual_base_x",
        "virtual_base_y",
        "virtual_base_theta",
        "virtual_base_center",
        "base_link",
    ]

    ARM_LINK = [
        "base_link_arm",
        "shoulder_link",
        "half_arm_1_link",
        "half_arm_2_link",
        "forearm_link",
        "spherical_wrist_1_link",
        "spherical_wrist_2_link",
        "bracelet_link",
        "end_effector_link",
        "camera_link",
        "camera_depth_frame",
        "camera_color_frame",
    ]

    END_EFFECTOR_LINK = [
        "robotiq_arg2f_base_link",
        "left_outer_knuckle",
        "right_outer_knuckle",
        "left_outer_finger",
        "right_outer_finger",
        "left_inner_finger",
        "right_inner_finger",
        "left_inner_finger_pad",
        "right_inner_finger_pad",
        "left_inner_knuckle",
        "right_inner_knuckle",
    ]

    END_EFFECTOR_LINK_VIS = [
        "robotiq_arg2f_base_link",
        "left_outer_knuckle",
        "right_outer_knuckle",
        "left_outer_finger",
        "right_outer_finger",
        "left_inner_finger",
        "right_inner_finger",
        "left_inner_knuckle",
        "right_inner_knuckle",
    ]

    DOF = 10
    BASE_DOF = 3
    ARM_DOF = 7
    urdf_path = RootPath.AGENT / "Mec_kinova" / "main.urdf"
    urdf_bullet_path = RootPath.AGENT / "Mec_kinova" / "main_bullet.urdf"
    pointcloud_cache = RootPath.AGENT / "Mec_kinova" / "pointcloud"

    def __init__(self):
        self.name = "Mec_kinova"
        self._urdf_path = str(MecKinova.urdf_path)
        self.urdf = URDF.load(self._urdf_path)
        self._scene = self.urdf.scene
    
    @property
    def trimesh(self) -> trimesh.Trimesh:
        self._scene = self.urdf.scene
        self._scene_trimesh = self._scene.dump(concatenate=True)
        vertex_colors = np.asarray(self._scene_trimesh.visual.to_color().vertex_colors)[:, :3] / 255
        # vertex_colors = np.expand_dims(np.array([0.118, 0.118, 0.118]), 0).repeat(vertex_colors.shape[0], 0)
        return trimesh.Trimesh(
            vertices=self._scene_trimesh.vertices,
            faces=self._scene_trimesh.faces,
            vertex_colors=vertex_colors,
            process=True,
        )
    
    def update_config(
        self,
        config: Union[List[int], np.ndarray],
    ):
        cfg = {}
        assert np.asarray(config).shape[0] == MecKinova.DOF, "Configuration dimension is wrong."
        for i in range(MecKinova.DOF):
            cfg[MecKinova.JOINTS_NAMES[i]] = config[i]
        self.urdf.update_cfg(cfg)
    
    def get_eff_pose(
        self,
        config: Optional[Union[List[int], np.ndarray]]=None,
    ) -> np.ndarray:
        """
        Get pose of the end-effector.
        """
        if config is None:
            return self.urdf.get_transform(
                frame_to="robotiq_arg2f_base_link", 
                frame_from="world"
            )
        else:
            self.update_config(config)
            return self.urdf.get_transform(
                frame_to="robotiq_arg2f_base_link", 
                frame_from="world"
            )
    
    def _simple_trimesh(self, link_trimesh, color:Optional[str]=None):
        if isinstance(link_trimesh.visual, trimesh.visual.color.ColorVisuals):
            vertex_colors = link_trimesh.visual.vertex_colors
        elif isinstance(link_trimesh.visual, trimesh.visual.texture.TextureVisuals):
            vertex_colors = link_trimesh.visual.to_color().vertex_colors
        else:
            raise Exception('Unsupported visual type')

        if color is not None:
            if color == 'red':
                vertex_colors = np.expand_dims(np.array([255, 0, 0]), 0).repeat(vertex_colors.shape[0], 0)
            elif color == 'green':
                vertex_colors = np.expand_dims(np.array([0, 255, 0]), 0).repeat(vertex_colors.shape[0], 0)
            elif color == 'blue':
                vertex_colors = np.expand_dims(np.array([0, 0, 255]), 0).repeat(vertex_colors.shape[0], 0)
            elif color == 'grey':
                vertex_colors = np.expand_dims(np.array([128, 128, 128]), 0).repeat(vertex_colors.shape[0], 0)
            elif color == 'black':
                vertex_colors = np.expand_dims(np.array([30, 30, 30]), 0).repeat(vertex_colors.shape[0], 0)
            else:
                raise Exception('Unsupported color type')
    
        return trimesh.Trimesh(
            vertices=link_trimesh.vertices,
            faces=link_trimesh.faces,
            vertex_colors=vertex_colors,
            process=True,
        )

    def get_eef_trimesh(self, config:Optional[Union[List[int], np.ndarray]]=None, color:Optional[str]=None):
        eef_trimesh = trimesh.Scene()
        if config is not None: self.update_config(config)

        eef_mesh = []
        for l_name in self.END_EFFECTOR_LINK_VIS:
            eef_mesh.append(self.get_link(l_name))

        eef_trimesh.add_geometry(eef_mesh)
        eef_trimesh = eef_trimesh.dump(concatenate=True)
        return self._simple_trimesh(eef_trimesh, color)
    
    def get_base_trimesh(self, config:Optional[Union[List[int], np.ndarray]]=None, color:Optional[str]=None):
        base_trimesh = trimesh.Scene()
        if config is not None: self.update_config(config)
        base_trimesh.add_geometry(self.get_link('base_link'))
        base_trimesh = base_trimesh.dump(concatenate=True)
        return self._simple_trimesh(base_trimesh, color)
    
    def sample(
        self, 
        config: Union[List[int], np.ndarray],
        num_sampled_points: int = 1024,
    ) -> np.ndarray:
        cfg = {}
        assert np.asarray(config).shape[0] == MecKinova.DOF
        for i in range(MecKinova.DOF):
            cfg[MecKinova.JOINTS_NAMES[i]] = config[i]

        self.urdf.update_cfg(cfg)
        agent_points, _ = trimesh.sample.sample_surface(self.trimesh, num_sampled_points)
        agent_points = np.asarray(agent_points)
        return agent_points
    
    def get_link(self, link_name: str):
        """
        Get the trimesh object in the agent frame.
        """
        link: Link = self.urdf.link_map[link_name]
        link_visuals = link.visuals[0]
        link_origin = link_visuals.origin
        link_geometry = link_visuals.geometry
        link_mesh = link_geometry.mesh
        link_filename = link_mesh.filename
        link_scale = link_mesh.scale
        if link_scale is None: link_scale = [1, 1, 1]
        link_file_path = os.path.join(
            str(RootPath.AGENT / self.name), link_filename
        )
        link_trimesh = trimesh.load(link_file_path, force="mesh")
        link_trimesh.apply_scale(link_scale)
        link_trimesh.apply_transform(
            self.urdf.get_transform(
                frame_to=link_name, 
                frame_from="world"
            ) @ link_origin
        )
        return link_trimesh
    
    def sample_eef_points(
        self, 
        config: Optional[Union[List[int], np.ndarray]]=None,
        eef_link_name: str="robotiq_arg2f_base_link",
        sample_num: int=1024,
    ) -> np.ndarray:
        if config is not None:
            self.update_config(config)
        eef_link_trimesh = self.get_link(eef_link_name)
        eef_link_points, _ = trimesh.sample.sample_surface(eef_link_trimesh, sample_num)
        return np.asarray(eef_link_points)

    @staticmethod
    def within_limits(cfg):
        # We have to add a small buffer because of float math
        return np.all(cfg >= MecKinova.JOINT_LIMITS[:, 0] - 1e-5) and np.all(cfg <= MecKinova.JOINT_LIMITS[:, 1] + 1e-5)
    
    @staticmethod
    def normalize_joints(
        batch_trajectory: Union[np.ndarray, torch.Tensor],
        limits: Tuple[float, float] = (-1, 1),
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalizes joint angles to be within a specified range according to the MecKinova's joint limits. 

        Arguements:
            batch_trajectory {Union[np.ndarray, torch.Tensor]} -- A batch of trajectories. Can have dims
                                                                  1) [10] if a single configuration
                                                                  2) [B, 10] if a batch of configurations
                                                                  3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The new limits to map to
        Returns:
            Union[np.ndarray, torch.Tensor] -- A tensor or numpy array with the same dimensions and type as the input       
        Raises:
            NotImplementedError -- Raises an error if another data type (e.g. a list) is passed in
        """
        if isinstance(batch_trajectory, torch.Tensor):
            return MecKinova._normalize_joints_torch(batch_trajectory, limits=limits)
        elif isinstance(batch_trajectory, np.ndarray):
            return MecKinova._normalize_joints_numpy(batch_trajectory, limits=limits)
        else:
            raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")
    
    @staticmethod
    def _normalize_joints_torch(
        batch_trajectory: torch.Tensor,
        limits: Tuple[float, float] = (-1, 1),
    ) -> torch.Tensor:
        """
        Normalizes joint angles to be within a specified range according to the MecKinova's joint limits. 

        Arguements:
            batch_trajectory {Union[np.ndarray, torch.Tensor]} -- A batch of trajectories. Can have dims
                                                                  1) [10] if a single configuration
                                                                  2) [B, 10] if a batch of configurations
                                                                  3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The new limits to map to
        Returns:
            Union[np.ndarray, torch.Tensor] -- A tensor or numpy array with the same dimensions and type as the input
        """
        assert isinstance(batch_trajectory, torch.Tensor)
        meckinova_limits = torch.as_tensor(MecKinova.JOINT_LIMITS).type_as(batch_trajectory)
        assert (
            (batch_trajectory.ndim == 1 and batch_trajectory.size(0) == MecKinova.DOF)
            or (batch_trajectory.ndim == 2 and batch_trajectory.size(1) == MecKinova.DOF)
            or (batch_trajectory.ndim == 3 and batch_trajectory.size(2) == MecKinova.DOF)
        )
        normalized = (batch_trajectory - meckinova_limits[:, 0]) / (
            meckinova_limits[:, 1] - meckinova_limits[:, 0]
        ) * (limits[1] - limits[0]) + limits[0]
        return normalized
    
    @staticmethod
    def _normalize_joints_numpy(
        batch_trajectory: np.ndarray,
        limits: Tuple[float, float] = (-1, 1),
    ) -> np.ndarray:
        """
        Normalizes joint angles to be within a specified range according to the MecKinova's joint limits. This is the numpy version. 

        Arguements:
            batch_trajectory {Union[np.ndarray, torch.Tensor]} -- A batch of trajectories. Can have dims
                                                                  1) [10] if a single configuration
                                                                  2) [B, 10] if a batch of configurations
                                                                  3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The new limits to map to
        Returns:
            Union[np.ndarray, torch.Tensor] -- A tensor or numpy array with the same dimensions and type as the input
        """
        assert isinstance(batch_trajectory, np.ndarray)
        meckinova_limits = MecKinova.JOINT_LIMITS
        assert (
            (batch_trajectory.ndim == 1 and batch_trajectory.shape[0] == MecKinova.DOF)
            or (batch_trajectory.ndim == 2 and batch_trajectory.shape[1] == MecKinova.DOF)
            or (batch_trajectory.ndim == 3 and batch_trajectory.shape[2] == MecKinova.DOF)
        )
        normalized = (batch_trajectory - meckinova_limits[:, 0]) / (
            meckinova_limits[:, 1] - meckinova_limits[:, 0]
        ) * (limits[1] - limits[0]) + limits[0]
        return normalized
    
    @staticmethod
    def unnormalize_joints(
        batch_trajectory: Union[np.ndarray, torch.Tensor],
        limits: Tuple[float, float] = (-1, 1),
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Unnormalizes joint angles from a specified range back into the MecKinova's joint limits.
        This is the inverse of `normalize_joints`.

        Arguements:
            batch_trajectory {Union[np.ndarray, torch.Tensor]} -- A batch of trajectories. Can have dims
                                                                  1) [10] if a single configuration
                                                                  2) [B, 10] if a batch of configurations
                                                                  3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The current limits to map to the joint limits        
        Returns:
            Union[np.ndarray, torch.Tensor] -- A tensor or numpy array with the same dimensions and type as the input        
        Raises:
            NotImplementedError -- Raises an error if another data type (e.g. a list) is passed in
        """
        if isinstance(batch_trajectory, torch.Tensor):
            return MecKinova._unnormalize_joints_torch(batch_trajectory, limits=limits)
        elif isinstance(batch_trajectory, np.ndarray):
            return MecKinova._unnormalize_joints_numpy(batch_trajectory, limits=limits)
        else:
            raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")
    
    @staticmethod
    def _unnormalize_joints_torch(
        batch_trajectory: torch.Tensor,
        limits: Tuple[float, float] = (-1, 1),
    ) -> torch.Tensor:
        """
        Unnormalizes joint angles from a specified range back into the MecKinova's joint limits.
        This is the torch version and the inverse of `_normalize_joints_torch`.

        Arguements:
            batch_trajectory {torch.Tensor} -- A batch of trajectories. Can have dims
                                               1) [10] if a single configuration
                                               2) [B, 10] if a batch of configurations
                                               3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The current limits to map to the joint limits        
        Returns:
            torch.Tensor -- A tensor with the same dimensions as the input
        """
        assert isinstance(batch_trajectory, torch.Tensor)
        meckinova_limits = torch.as_tensor(MecKinova.JOINT_LIMITS).type_as(batch_trajectory)
        assert (
            (batch_trajectory.ndim == 1 and batch_trajectory.size(0) == MecKinova.DOF)
            or (batch_trajectory.ndim == 2 and batch_trajectory.size(1) == MecKinova.DOF)
            or (batch_trajectory.ndim == 3 and batch_trajectory.size(2) == MecKinova.DOF)
        )
        # assert torch.all(batch_trajectory >= limits[0])
        # assert torch.all(batch_trajectory <= limits[1])
        meckinova_limit_range = meckinova_limits[:, 1] - meckinova_limits[:, 0]
        meckinova_lower_limit = meckinova_limits[:, 0]
        for _ in range(batch_trajectory.ndim - 1):
            meckinova_limit_range = meckinova_limit_range.unsqueeze(0)
            meckinova_lower_limit = meckinova_lower_limit.unsqueeze(0)
        unnormalized = (batch_trajectory - limits[0]) * meckinova_limit_range / (
            limits[1] - limits[0]
        ) + meckinova_lower_limit
        return unnormalized
    
    @staticmethod
    def _unnormalize_joints_numpy(
        batch_trajectory: np.ndarray,
        limits: Tuple[float, float] = (-1, 1),
    ) -> np.ndarray:
        """
        Unnormalizes joint angles from a specified range back into the MecKinova's joint limits.
        This is the torch version and the inverse of `_normalize_joints_numpy`.

        Arguements:
            batch_trajectory {torch.Tensor} -- A batch of trajectories. Can have dims
                                               1) [10] if a single configuration
                                               2) [B, 10] if a batch of configurations
                                               3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The current limits to map to the joint limits       
        Returns:
            torch.Tensor -- A tensor with the same dimensions as the input
        """
        assert isinstance(batch_trajectory, np.ndarray)
        meckinova_limits = MecKinova.JOINT_LIMITS
        assert (
            (batch_trajectory.ndim == 1 and batch_trajectory.shape[0] == MecKinova.DOF)
            or (batch_trajectory.ndim == 2 and batch_trajectory.shape[1] == MecKinova.DOF)
            or (batch_trajectory.ndim == 3 and batch_trajectory.shape[2] == MecKinova.DOF)
        )
        # assert np.all(batch_trajectory >= limits[0])
        # assert np.all(batch_trajectory <= limits[1])
        meckinova_limit_range = meckinova_limits[:, 1] - meckinova_limits[:, 0]
        meckinova_lower_limit = meckinova_limits[:, 0]
        for _ in range(batch_trajectory.ndim - 1):
            meckinova_limit_range = meckinova_limit_range[np.newaxis, ...]
            meckinova_lower_limit = meckinova_lower_limit[np.newaxis, ...]
        unnormalized = (batch_trajectory - limits[0]) * meckinova_limit_range / (
            limits[1] - limits[0]
        ) + meckinova_lower_limit
        return unnormalized
    
    @staticmethod
    def normalize_actions(
        batch_delta_trajectory: Union[np.ndarray, torch.Tensor],
        limits: Tuple[float, float] = (-1, 1),
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalizes delta joint angles to be within a specified range according to the MecKinova's delta joint limits. 

        Arguements:
            batch_delta_trajectory {Union[np.ndarray, torch.Tensor]} -- A batch of delta trajectories. Can have dims
                                                                  1) [10] if a single configuration
                                                                  2) [B, 10] if a batch of configurations
                                                                  3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The new limits to map to
        Returns:
            Union[np.ndarray, torch.Tensor] -- A tensor or numpy array with the same dimensions and type as the input       
        Raises:
            NotImplementedError -- Raises an error if another data type (e.g. a list) is passed in
        """
        if isinstance(batch_delta_trajectory, torch.Tensor):
            return MecKinova._normalize_actions_torch(batch_delta_trajectory, limits=limits)
        elif isinstance(batch_delta_trajectory, np.ndarray):
            return MecKinova._normalize_actions_numpy(batch_delta_trajectory, limits=limits)
        else:
            raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")
    
    @staticmethod
    def _normalize_actions_torch(
        batch_delta_trajectory: torch.Tensor,
        limits: Tuple[float, float] = (-1, 1),
    ) -> torch.Tensor:
        """
        Normalizes delta joint angles to be within a specified range according to the MecKinova's delta joint limits. 

        Arguements:
            batch_delta_trajectory {Union[np.ndarray, torch.Tensor]} -- A batch of delta trajectories. Can have dims
                                                                  1) [10] if a single configuration
                                                                  2) [B, 10] if a batch of configurations
                                                                  3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The new limits to map to
        Returns:
            Union[np.ndarray, torch.Tensor] -- A tensor or numpy array with the same dimensions and type as the input
        """
        assert isinstance(batch_delta_trajectory, torch.Tensor)
        meckinova_action_limits = torch.as_tensor(MecKinova.ACTION_LIMITS).type_as(batch_delta_trajectory)
        assert (
            (batch_delta_trajectory.ndim == 1 and batch_delta_trajectory.size(0) == MecKinova.DOF)
            or (batch_delta_trajectory.ndim == 2 and batch_delta_trajectory.size(1) == MecKinova.DOF)
            or (batch_delta_trajectory.ndim == 3 and batch_delta_trajectory.size(2) == MecKinova.DOF)
        )
        normalized = (batch_delta_trajectory - meckinova_action_limits[:, 0]) / (
            meckinova_action_limits[:, 1] - meckinova_action_limits[:, 0]
        ) * (limits[1] - limits[0]) + limits[0]
        return normalized
    
    @staticmethod
    def _normalize_actions_numpy(
        batch_delta_trajectory: np.ndarray,
        limits: Tuple[float, float] = (-1, 1),
    ) -> np.ndarray:
        """
        Normalizes delta joint angles to be within a specified range according to the MecKinova's delta joint limits. 
        This is the numpy version. 

        Arguements:
            batch_delta_trajectory {Union[np.ndarray, torch.Tensor]} -- A batch of delta trajectories. Can have dims
                                                                  1) [10] if a single configuration
                                                                  2) [B, 10] if a batch of configurations
                                                                  3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The new limits to map to
        Returns:
            Union[np.ndarray, torch.Tensor] -- A tensor or numpy array with the same dimensions and type as the input
        """
        assert isinstance(batch_delta_trajectory, np.ndarray)
        meckinova_action_limits = MecKinova.ACTION_LIMITS
        assert (
            (batch_delta_trajectory.ndim == 1 and batch_delta_trajectory.shape[0] == MecKinova.DOF)
            or (batch_delta_trajectory.ndim == 2 and batch_delta_trajectory.shape[1] == MecKinova.DOF)
            or (batch_delta_trajectory.ndim == 3 and batch_delta_trajectory.shape[2] == MecKinova.DOF)
        )
        normalized = (batch_delta_trajectory - meckinova_action_limits[:, 0]) / (
            meckinova_action_limits[:, 1] - meckinova_action_limits[:, 0]
        ) * (limits[1] - limits[0]) + limits[0]
        return normalized
    
    @staticmethod
    def unnormalize_actions(
        batch_delta_trajectory: Union[np.ndarray, torch.Tensor],
        limits: Tuple[float, float] = (-1, 1),
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Unnormalizes delta joint angles from a specified range back into the MecKinova's delta joint limits.
        This is the inverse of `normalize_joints`.

        Arguements:
            batch_delta_trajectory {Union[np.ndarray, torch.Tensor]} -- A batch of delta_trajectories. Can have dims
                                                                  1) [10] if a single configuration
                                                                  2) [B, 10] if a batch of configurations
                                                                  3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The current limits to map to the joint limits        
        Returns:
            Union[np.ndarray, torch.Tensor] -- A tensor or numpy array with the same dimensions and type as the input        
        Raises:
            NotImplementedError -- Raises an error if another data type (e.g. a list) is passed in
        """
        if isinstance(batch_delta_trajectory, torch.Tensor):
            return MecKinova._unnormalize_actions_torch(batch_delta_trajectory, limits=limits)
        elif isinstance(batch_delta_trajectory, np.ndarray):
            return MecKinova._unnormalize_actions_numpy(batch_delta_trajectory, limits=limits)
        else:
            raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")
    
    @staticmethod
    def _unnormalize_actions_torch(
        batch_delta_trajectory: torch.Tensor,
        limits: Tuple[float, float] = (-1, 1),
    ) -> torch.Tensor:
        """
        Unnormalizes delta joint angles from a specified range back into the MecKinova's delta joint limits.
        This is the torch version and the inverse of `_normalize_joints_torch`.

        Arguements:
            batch_delta_trajectory {torch.Tensor} -- A batch of delta trajectories. Can have dims
                                               1) [10] if a single configuration
                                               2) [B, 10] if a batch of configurations
                                               3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The current limits to map to the joint limits        
        Returns:
            torch.Tensor -- A tensor with the same dimensions as the input
        """
        assert isinstance(batch_delta_trajectory, torch.Tensor)
        meckinova_action_limits = torch.as_tensor(MecKinova.ACTION_LIMITS).type_as(batch_delta_trajectory)
        assert (
            (batch_delta_trajectory.ndim == 1 and batch_delta_trajectory.size(0) == MecKinova.DOF)
            or (batch_delta_trajectory.ndim == 2 and batch_delta_trajectory.size(1) == MecKinova.DOF)
            or (batch_delta_trajectory.ndim == 3 and batch_delta_trajectory.size(2) == MecKinova.DOF)
        )
        assert torch.all(batch_delta_trajectory >= limits[0])
        assert torch.all(batch_delta_trajectory <= limits[1])
        meckinova_limit_range = meckinova_action_limits[:, 1] - meckinova_action_limits[:, 0]
        meckinova_lower_limit = meckinova_action_limits[:, 0]
        for _ in range(batch_delta_trajectory.ndim - 1):
            meckinova_limit_range = meckinova_limit_range.unsqueeze(0)
            meckinova_lower_limit = meckinova_lower_limit.unsqueeze(0)
        unnormalized = (batch_delta_trajectory - limits[0]) * meckinova_limit_range / (
            limits[1] - limits[0]
        ) + meckinova_lower_limit
        return unnormalized
    
    @staticmethod
    def _unnormalize_actions_numpy(
        batch_delta_trajectory: np.ndarray,
        limits: Tuple[float, float] = (-1, 1),
    ) -> np.ndarray:
        """
        Unnormalizes delta joint angles from a specified range back into the MecKinova's delta joint limits.
        This is the torch version and the inverse of `_normalize_joints_numpy`.

        Arguements:
            batch_delta_trajectory {torch.Tensor} -- A batch of delta trajectories. Can have dims
                                               1) [10] if a single configuration
                                               2) [B, 10] if a batch of configurations
                                               3) [B, T, 10] if a batched time-series of configurations
            limits {Tuple[float, float]} -- The current limits to map to the joint limits       
        Returns:
            torch.Tensor -- A tensor with the same dimensions as the input
        """
        assert isinstance(batch_delta_trajectory, np.ndarray)
        meckinova_action_limits = MecKinova.ACTION_LIMITS
        assert (
            (batch_delta_trajectory.ndim == 1 and batch_delta_trajectory.shape[0] == MecKinova.DOF)
            or (batch_delta_trajectory.ndim == 2 and batch_delta_trajectory.shape[1] == MecKinova.DOF)
            or (batch_delta_trajectory.ndim == 3 and batch_delta_trajectory.shape[2] == MecKinova.DOF)
        )
        assert np.all(batch_delta_trajectory >= limits[0])
        assert np.all(batch_delta_trajectory <= limits[1])
        meckinova_limit_range = meckinova_action_limits[:, 1] - meckinova_action_limits[:, 0]
        meckinova_lower_limit = meckinova_action_limits[:, 0]
        for _ in range(batch_delta_trajectory.ndim - 1):
            meckinova_limit_range = meckinova_limit_range[np.newaxis, ...]
            meckinova_lower_limit = meckinova_lower_limit[np.newaxis, ...]
        unnormalized = (batch_delta_trajectory - limits[0]) * meckinova_limit_range / (
            limits[1] - limits[0]
        ) + meckinova_lower_limit
        return unnormalized