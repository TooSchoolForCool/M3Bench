import logging
import os
import torch
import numpy as np
import trimesh
from env.agent.mec_kinova import MecKinova
import torch
from utils.torch_urdf import TorchURDF
from geometrout.primitive import Sphere
from utils.transform import transform_pointcloud_torch

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