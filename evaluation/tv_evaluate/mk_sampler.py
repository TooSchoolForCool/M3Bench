import logging
import os
from pathlib import Path
import torch
import numpy as np
import trimesh
import os
import torch
import trimesh
import numpy as np
from typing import List, Optional, Union, Tuple
from math import pi
from yourdfpy import URDF, Link
import torch
from urchin import (
    URDF,
    URDFTypeWithMesh,
    Joint,
    Link,
    Transmission,
    Material,
    Inertial,
    Visual,
    Collision,
)
from urchin.utils import parse_origin
from collections import OrderedDict
from lxml import etree as ET
import os
import torch
import numpy as np
from geometrout.primitive import Sphere


ASSETS_VERSION = "v0.0.6"

class RootPath:
    VKC_DEPS = Path("/home/robot-learning/0_WorkSpace/9_MGeNets/env")
    AGENT = VKC_DEPS / "tongverse_agents" / "agent"
    SCENE = VKC_DEPS / "physcene"

class MecKinovaSampler:
    """
    This class allows for fast pointcloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces pointclouds for each configuration by running fk on a subsample
    of the per-link pointclouds that are established at initialization.

    """
    # ignore_link = ["left_inner_finger_pad", "right_inner_finger_pad"]

    def __init__(
        self,
        device,
        num_fixed_points=None,
        use_cache=False,
        max_points=4096,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.num_fixed_points = num_fixed_points
        self._init_internal_(device, use_cache, max_points)

    def _init_internal_(self, device, use_cache, max_points):
        self.device = device
        self.max_points = max_points
        self.robot = TorchURDF.load(
            str(MecKinova.urdf_path), lazy_load_meshes=True, device=device
        )
        self.links = [l for l in self.robot.links if len(l.visuals)]
        self.end_effector_links = [l for l in self.links if l.name in MecKinova.END_EFFECTOR_LINK]
        # self.mesh_links = [l for l in self.links if l.visuals[0].geometry.mesh != None]
        if use_cache and self._init_from_cache_(device):
            return

        meshes = []
        for l in self.links: 
            if l.visuals[0].geometry.mesh is None:
                if l.visuals[0].geometry.box:
                    box = l.visuals[0].geometry.box
                    cuboid = trimesh.creation.box(box.size)
                    meshes.append(cuboid)
            else:
                mesh = l.visuals[0].geometry.mesh
                filename = mesh.filename
                scale = (1.0, 1.0, 1.0) if not isinstance(mesh.scale, np.ndarray) else mesh.scale
                filepath = MecKinova.urdf_path.parent / filename
                tmesh = trimesh.load(filepath, force="mesh")
                tmesh.apply_scale(scale)
                meshes.append(tmesh)

        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        if self.num_fixed_points is not None:
            num_points = np.round(
                self.num_fixed_points * np.array(areas) / np.sum(areas)
            )
            num_points[0] += self.num_fixed_points - np.sum(num_points)
            assert np.sum(num_points) == self.num_fixed_points
        else:
            num_points = np.round(max_points * np.array(areas) / np.sum(areas))
        self.points = {}
        for ii in range(len(meshes)):
            pc = trimesh.sample.sample_surface(meshes[ii], int(num_points[ii]))[0]
            self.points[self.links[ii].name] = torch.as_tensor(
                pc, device=device
            ).unsqueeze(0)

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally
        if use_cache:
            points_to_save = {
                k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.points.items()
            }
            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)
    
    def _get_cache_file_name_(self):
        if self.num_fixed_points is not None:
            return (
                MecKinova.pointcloud_cache
                / f"fixed_point_cloud_{self.num_fixed_points}.npy"
            )
        else:
            return MecKinova.pointcloud_cache / "full_point_cloud.npy"

    def _init_from_cache_(self, device):
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = {
            key: torch.as_tensor(pc, device=device).unsqueeze(0)
            for key, pc in points.item().items()
        }
        return True
    
    def sample(self, config, num_points=None):
        """
        Samples points from the surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of  actuated joints.
                 For example, if using the MecKinova, M is 10
        num_points : Number of points desired

        Returns
        -------
        N x num points x 3 pointcloud of robot points

        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None) 
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            fk_transforms[l.name] = values[idx]
            pc = transform_pointcloud_torch(
                self.points[l.name]
                .float()
                .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                fk_transforms[l.name],
                in_place=True,
            )
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def sample_base(self, config, num_points=None):
        """
        Samples points from the base surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of  actuated joints.
                 For example, if using the MecKinova, M is 10
        num_points : Number of points desired

        Returns
        -------
        N x num points x 3 pointcloud of robot base points

        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None) 
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            if l.name in MecKinova.BASE_LINK:
                fk_transforms[l.name] = values[idx]
                pc = transform_pointcloud_torch(
                    self.points[l.name]
                    .float()
                    .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                    fk_transforms[l.name],
                    in_place=True,
                )
                fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def sample_arm(self, config, num_points=None):
        """
        Samples points from the arm surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of  actuated joints.
                 For example, if using the MecKinova, M is 10
        num_points : Number of points desired

        Returns
        -------
        N x num points x 3 pointcloud of robot arm points

        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None) 
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            if l.name in MecKinova.ARM_LINK:
                fk_transforms[l.name] = values[idx]
                pc = transform_pointcloud_torch(
                    self.points[l.name]
                    .float()
                    .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                    fk_transforms[l.name],
                    in_place=True,
                )
                fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def sample_gripper(self, config, num_points=None):
        """
        Samples points from the gripper surface of the robot by calling fk.
        It does the same thing as sample_end_effector, except that it takes 
        points from the cache and performs coordinate transformations.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of  actuated joints.
                 For example, if using the MecKinova, M is 10
        num_points : Number of points desired

        Returns
        -------
        N x num points x 3 pointcloud of robot gripper points

        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None) 
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            if l.name in MecKinova.END_EFFECTOR_LINK:
                fk_transforms[l.name] = values[idx]
                pc = transform_pointcloud_torch(
                    self.points[l.name]
                    .float()
                    .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                    fk_transforms[l.name],
                    in_place=True,
                )
                fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def end_effector_pose(self, config, frame="end_effector_link") -> torch.Tensor:
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        return fk[frame]
    
    def _get_eef_cache_file_name_(self, eef_points_num):
        return (
            MecKinova.pointcloud_cache
            / f"eef_point_cloud_{eef_points_num}.npy"
        )

    def _init_from_eef_cache_(self, device, eef_points_num):
        eef_file_name = self._get_eef_cache_file_name_(eef_points_num)
        if not os.path.exists(eef_file_name):
            return False
        eef_points = np.load(
            eef_file_name,
            allow_pickle=True,
        )
        self.eef_points = {
            key: torch.as_tensor(pc, device=device).unsqueeze(0)
            for key, pc in eef_points.item().items()
        }
        return True
    
    def sample_end_effector(self, config, sample_points=512, use_cache=False):
        """
        End Effector PointClouds Sample.
        """
        self.eef_points = {}
        if use_cache and self._init_from_eef_cache_(self.device, sample_points):
            pass
        else:
            end_effector_meshes = []
            for l in self.end_effector_links: 
                if l.visuals[0].geometry.mesh is None:
                    if l.visuals[0].geometry.box:
                        box = l.visuals[0].geometry.box
                        cuboid = trimesh.creation.box(box.size)
                        end_effector_meshes.append(cuboid)
                else:
                    mesh = l.visuals[0].geometry.mesh
                    filename = mesh.filename
                    scale = (1.0, 1.0, 1.0) if not isinstance(mesh.scale, np.ndarray) else mesh.scale
                    filepath = MecKinova.urdf_path.parent / filename
                    tmesh = trimesh.load(filepath, force="mesh")
                    tmesh.apply_scale(scale)
                    end_effector_meshes.append(tmesh)
            
            areas = [mesh.bounding_box_oriented.area for mesh in end_effector_meshes]

            num_points = np.round(
                sample_points * np.array(areas) / np.sum(areas)
            )
            num_points[0] += sample_points - np.sum(num_points)
            assert np.sum(num_points) == sample_points

            # sample Points
            for i in range(len(end_effector_meshes)):
                pc = trimesh.sample.sample_surface(end_effector_meshes[i], int(num_points[i]))[0]
                self.eef_points[self.end_effector_links[i].name] = torch.as_tensor(
                    pc, device=self.device
                ).unsqueeze(0)

            # save points
            if use_cache:
                points_to_save = {
                    k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.eef_points.items()
                }
                eef_file_name = self._get_eef_cache_file_name_(sample_points)
                print(f"Saving new file to cache: {eef_file_name}")
                np.save(eef_file_name, points_to_save)
        
        # transform
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())

        assert len(self.links) == len(values)
        ef_transforms = {}
        ef_points = []
        for idx, l in enumerate(self.links):
            if l in self.end_effector_links:
                ef_transforms[l.name] = values[idx]
                pc = transform_pointcloud_torch(
                    self.eef_points[l.name]
                    .float()
                    .repeat((ef_transforms[l.name].shape[0], 1, 1)),
                    ef_transforms[l.name],
                    in_place=True,
                )
                ef_points.append(pc)
        pc = torch.cat(ef_points, dim=1)
        return pc


class MecKinovaCollisionSampler:
    def __init__(
        self,
        device,
        margin=0.0,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.robot = TorchURDF.load(
            str(MecKinova.urdf_path), lazy_load_meshes=True, device=device
        )
        self.spheres = []
        for radius, point_set in MecKinova.SPHERES:
            sphere_centers = {
                k: torch.as_tensor(v).to(device) for k, v in point_set.items()
            }
            if not len(sphere_centers):
                continue
            self.spheres.append(
                (
                    radius + margin,
                    sphere_centers,
                )
            )
        
        all_spheres = {}
        for radius, point_set in MecKinova.SPHERES:
            for link_name, centers in point_set.items():
                for c in centers:
                    all_spheres[link_name] = all_spheres.get(link_name, []) + [
                        Sphere(np.asarray(c), radius + margin)
                    ]
        
        total_points = 10000
        surface_scalar_sum = sum(
            [sum([s.radius ** 2 for s in v]) for v in all_spheres.values()]
        )
        surface_scalar = total_points / surface_scalar_sum
        self.link_points = {}
        for link_name, spheres in all_spheres.items():
            self.link_points[link_name] = torch.as_tensor(
                np.concatenate(
                    [
                        s.sample_surface(int(surface_scalar * s.radius ** 2))
                        for s in spheres
                    ],
                    axis=0,
                ),
                device=device,
            )
    
    def sample(self, config, n):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        pointcloud = []
        for link_name, points in self.link_points.items():
            pc = transform_pointcloud_torch(
                points.float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=True,
            )
            pointcloud.append(pc)
        pc = torch.cat(pointcloud, dim=1)
        return pc[:, np.random.choice(pc.shape[1], n, replace=False), :]

    def compute_spheres(self, config):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg  = config
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        points = []
        for radius, spheres in self.spheres:
            fk_points = []
            for link_name in spheres:
                pc = transform_pointcloud_torch(
                    spheres[link_name]
                    .type_as(cfg)
                    .repeat((fk[link_name].shape[0], 1, 1)),
                    fk[link_name].type_as(cfg),
                    in_place=True,
                )
                fk_points.append(pc)
            points.append((radius, torch.cat(fk_points, dim=1)))
        return points
    

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
        return self._scene.dump(concatenate=True)
    
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

def configure_origin(value, device=None):
    """Convert a value into a 4x4 transform matrix.
    Parameters
    ----------
    value : None, (6,) float, or (4,4) float
        The value to turn into the matrix.
        If (6,), interpreted as xyzrpy coordinates.
    Returns
    -------
    matrix : (4,4) float or None
        The created matrix.
    """
    assert isinstance(
        value, torch.Tensor
    ), "Invalid type for origin, expect 4x4 torch tensor"
    assert value.shape == (4, 4)
    return value.to(device)


class TorchVisual(Visual):
    def __init__(self, geometry, name=None, origin=None, material=None, device=None):
        self.device = device
        super().__init__(geometry, name, origin, material)

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value, self.device)

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        kwargs = cls._parse(node, path, lazy_load_meshes)
        kwargs["origin"] = torch.tensor(parse_origin(node))
        kwargs["device"] = device
        return TorchVisual(**kwargs)


class TorchCollision(Collision):
    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        kwargs = cls._parse(node, path, lazy_load_meshes)
        kwargs["origin"] = parse_origin(node)
        return TorchCollision(**kwargs)


class TorchLink(Link):
    _ELEMENTS = {
        "inertial": (Inertial, False, False),
        "visuals": (TorchVisual, False, True),
        "collisions": (TorchCollision, False, True),
    }

    def __init__(self, name, inertial, visuals, collisions, device=None):
        self.device = device
        super().__init__(name, inertial, visuals, collisions)

    @classmethod
    def _parse_simple_elements(cls, node, path, lazy_load_meshes, device):
        """Parse all elements in the _ELEMENTS array from the children of
        this node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs = {}
        for a in cls._ELEMENTS:
            t, r, m = cls._ELEMENTS[a]
            if not m:
                v = node.find(t._TAG)
                if r or v is not None:
                    if issubclass(t, URDFTypeWithMesh):
                        v = t._from_xml(v, path, lazy_load_meshes)
                    else:
                        v = t._from_xml(v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:
                    raise ValueError(
                        "Missing required subelement(s) of type {} when "
                        "parsing an object of type {}".format(t.__name__, cls.__name__)
                    )
                if issubclass(t, URDFTypeWithMesh):
                    v = [t._from_xml(n, path, lazy_load_meshes, device) for n in vs]
                else:
                    v = [t._from_xml(n, path, device) for n in vs]
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse(cls, node, path, lazy_load_meshes, device):
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from names to Python classes created from the attributes
            and elements in the class arrays.
        """
        kwargs = cls._parse_simple_attribs(node)
        kwargs.update(cls._parse_simple_elements(node, path, lazy_load_meshes, device))
        return kwargs

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        """Create an instance of this class from an XML node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        obj : :class:`URDFType`
            An instance of this class parsed from the node.
        """
        return cls(**cls._parse(node, path, lazy_load_meshes, device))


class TorchJoint(Joint):
    def __init__(
        self,
        name,
        joint_type,
        parent,
        child,
        axis=None,
        origin=None,
        limit=None,
        dynamics=None,
        safety_controller=None,
        calibration=None,
        mimic=None,
        device=None,
    ):
        self.device = device
        super().__init__(
            name,
            joint_type,
            parent,
            child,
            axis,
            origin,
            limit,
            dynamics,
            safety_controller,
            calibration,
            mimic,
        )

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value, device=self.device)

    @property
    def axis(self):
        """(3,) float : The joint axis in the joint frame."""
        return self._axis

    @axis.setter
    def axis(self, value):
        if value is None:
            value = torch.as_tensor([1.0, 0.0, 0.0], device=self.device)
        elif isinstance(value, torch.Tensor):
            assert value.shape == (3,), "Invalid shape for axis, should be (3,)"
            value = value.to(self.device)
            value = value / torch.norm(value)
        else:
            value = torch.as_tensor(value, device=self.device)
            if value.shape != (3,):
                raise ValueError("Invalid shape for axis, should be (3,)")
            value = value / torch.norm(value)
        self._axis = value

    @classmethod
    def _from_xml(cls, node, path, device):
        kwargs = cls._parse(node, path)
        kwargs["joint_type"] = str(node.attrib["type"])
        kwargs["parent"] = node.find("parent").attrib["link"]
        kwargs["child"] = node.find("child").attrib["link"]
        axis = node.find("axis")
        if axis is not None:
            axis = torch.as_tensor(
                np.fromstring(axis.attrib["xyz"], sep=" "),
            )
        kwargs["axis"] = axis
        kwargs["origin"] = torch.tensor(parse_origin(node))
        kwargs["device"] = device

        return TorchJoint(**kwargs)

    def _rotation_matrices(self, angles, axis):
        """Compute rotation matrices from angle/axis representations.
        Parameters
        ----------
        angles : (n,) float
            The angles.
        axis : (3,) float
            The axis.
        Returns
        -------
        rots : (n,4,4)
            The rotation matrices
        """
        axis = axis / torch.norm(axis)
        sina = torch.sin(angles)
        cosa = torch.cos(angles)
        M = torch.eye(4, device=self.device).repeat((len(angles), 1, 1))
        M[:, 0, 0] = cosa
        M[:, 1, 1] = cosa
        M[:, 2, 2] = cosa
        M[:, :3, :3] += (
            torch.ger(axis, axis).repeat((len(angles), 1, 1))
            * (1.0 - cosa)[:, np.newaxis, np.newaxis]
        )
        M[:, :3, :3] += (
            torch.tensor(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                device=self.device,
            ).repeat((len(angles), 1, 1))
            * sina[:, np.newaxis, np.newaxis]
        )
        return M

    def get_child_poses(self, cfg, n_cfgs):
        """Computes the child pose relative to a parent pose for a given set of
        configuration values.
        Parameters
        ----------
        cfg : (n,) float or None
            The configuration values for this joint. They are interpreted
            based on the joint type as follows:
            - ``fixed`` - not used.
            - ``prismatic`` - a translation along the axis in meters.
            - ``revolute`` - a rotation about the axis in radians.
            - ``continuous`` - a rotation about the axis in radians.
            - ``planar`` - Not implemented.
            - ``floating`` - Not implemented.
            If ``cfg`` is ``None``, then this just returns the joint pose.
        Returns
        -------
        poses : (n,4,4) float
            The poses of the child relative to the parent.
        """
        if cfg is None:
            return self.origin.repeat((n_cfgs, 1, 1))
        elif self.joint_type == "fixed":
            return self.origin.repeat((n_cfgs, 1, 1))
        elif self.joint_type in ["revolute", "continuous"]:
            if cfg is None:
                cfg = torch.zeros(n_cfgs)
            return torch.matmul(
                self.origin.type_as(cfg),
                self._rotation_matrices(cfg, self.axis).type_as(cfg),
            )
        elif self.joint_type == "prismatic":
            if cfg is None:
                cfg = torch.zeros(n_cfgs)
            translation = torch.eye(4, device=self.device).repeat((n_cfgs, 1, 1))
            translation[:, :3, 3] = self.axis * cfg[:, np.newaxis]
            return torch.matmul(self.origin.type_as(cfg), translation.type_as(cfg))
        elif self.joint_type == "planar":
            raise NotImplementedError()
        elif self.joint_type == "floating":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid configuration")

class TorchURDF(URDF):

    _ELEMENTS = {
        "links": (TorchLink, True, True),
        "joints": (TorchJoint, False, True),
        "transmissions": (Transmission, False, True),
        "materials": (Material, False, True),
    }

    def __init__(
        self,
        name,
        links,
        joints=None,
        transmissions=None,
        materials=None,
        other_xml=None,
        device=None,
    ):
        self.device = device
        super().__init__(name, links, joints, transmissions, materials, other_xml)

    @staticmethod
    def load(file_obj, lazy_load_meshes=True, device=None):
        """Load a URDF from a file.
        Parameters
        ----------
        file_obj : str or file-like object
            The file to load the URDF from. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.
        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        if isinstance(file_obj, str):
            if os.path.isfile(file_obj):
                parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
                tree = ET.parse(file_obj, parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError("{} is not a file".format(file_obj))
        else:
            parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return TorchURDF._from_xml(node, path, lazy_load_meshes, device)

    @classmethod
    def _parse_simple_elements(cls, node, path, lazy_load_meshes, device):
        """Parse all elements in the _ELEMENTS array from the children of
        this node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs = {}
        for a in cls._ELEMENTS:
            t, r, m = cls._ELEMENTS[a]
            if not m:
                v = node.find(t._TAG)
                if r or v is not None:
                    if issubclass(t, URDFTypeWithMesh):
                        v = t._from_xml(v, path, lazy_load_meshes)
                    else:
                        v = t._from_xml(v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:
                    raise ValueError(
                        "Missing required subelement(s) of type {} when "
                        "parsing an object of type {}".format(t.__name__, cls.__name__)
                    )
                if issubclass(t, URDFTypeWithMesh):
                    v = [t._from_xml(n, path, lazy_load_meshes, device) for n in vs]
                else:
                    v = [t._from_xml(n, path, device) for n in vs]
            kwargs[a] = v
        return kwargs


    @classmethod
    def _parse(cls, node, path, lazy_load_meshes, device):
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from names to Python classes created from the attributes
            and elements in the class arrays.
        """
        kwargs = cls._parse_simple_attribs(node)
        kwargs.update(cls._parse_simple_elements(node, path, lazy_load_meshes, device))
        return kwargs

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        valid_tags = set(["joint", "link", "transmission", "material"])
        kwargs = cls._parse(node, path, lazy_load_meshes, device)

        extra_xml_node = ET.Element("extra")
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)

        data = ET.tostring(extra_xml_node)
        kwargs["other_xml"] = data
        kwargs["device"] = device
        return cls(**kwargs)

    def _process_cfgs(self, cfgs):
        """Process a list of joint configurations into a dictionary mapping joints to
        configuration values.
        This should result in a dict mapping each joint to a list of cfg values, one
        per joint.
        """
        joint_cfg = {}
        assert isinstance(cfgs, torch.Tensor), "Incorrectly formatted config array"
        n_cfgs = len(cfgs)
        for i, j in enumerate(self.actuated_joints):
            joint_cfg[j] = cfgs[:, i]

        return joint_cfg, n_cfgs

    def link_fk(self, cfg=None, link=None, links=None, use_names=False):
        raise NotImplementedError("Not implemented")

    def link_fk_batch(self, cfgs=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics in a batch.
        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            link names rather than the links themselves.
        Returns
        -------
        fk : dict or (n,4,4) float
            A map from links to a (n,4,4) vector of homogenous transform matrices that
            position the links relative to the base link's frame
        """
        joint_cfgs, n_cfgs = self._process_cfgs(cfgs)

        # Process link set
        link_set = self.links

        # Compute FK mapping each link to a vector of matrices, one matrix per cfg
        fk = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            poses = torch.eye(4, device=self.device).repeat((n_cfgs, 1, 1))
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)["joint"]

                cfg_vals = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfgs:
                        cfg_vals = joint_cfgs[mimic_joint]
                        cfg_vals = (
                            joint.mimic.multiplier * cfg_vals + joint.mimic.offset
                        )
                elif joint in joint_cfgs:
                    cfg_vals = joint_cfgs[joint]

                child_poses = joint.get_child_poses(cfg_vals, n_cfgs)
                poses = torch.matmul(child_poses, poses.type_as(child_poses))

                if parent in fk:
                    poses = torch.matmul(fk[parent], poses.type_as(fk[parent]))
                    break
            fk[lnk] = poses

        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def visual_geometry_fk_batch(self, cfgs=None):
        """Computes the poses of the URDF's visual geometries using fk.
        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only geometries from these links will be in the returned map.
            If not specified, all links are returned.
        Returns
        -------
        fk : dict
            A map from :class:`Geometry` objects that are part of the visual
            elements of the specified links to the 4x4 homogenous transform
            matrices that position them relative to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs)

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                fk[visual.geometry] = torch.matmul(
                    lfk[link], visual.origin.type_as(lfk[link])
                )
        return fk

def transform_pointcloud_torch(pc, transformation_matrix, in_place=True):
    """
    Parameters
    ----------
    pc: A pytorch tensor pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        M could be some additional mask dimensions or whatever, but the
        3 are x-y-z
    transformation_matrix: A 4x4 homography

    Returns
    -------
    Mutates the pointcloud in place and transforms x, y, z according the homography
    """
    assert isinstance(pc, torch.Tensor)
    assert type(pc) == type(transformation_matrix)
    assert pc.ndim == transformation_matrix.ndim
    if pc.ndim == 3:
        N, M = 1, 2
    elif pc.ndim == 2:
        N, M = 0, 1
    else:
        raise Exception("Pointcloud must have dimension Nx3 or BxNx3")
    xyz = pc[..., :3]
    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    homogeneous_xyz = torch.cat((xyz, torch.ones(ones_dim, device=xyz.device)), dim=M)
    transformed_xyz = torch.matmul(
        transformation_matrix, homogeneous_xyz.transpose(N, M)
    )
    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
        return pc
    return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)