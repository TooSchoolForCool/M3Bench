import copy
import shutil
from trimesh import transform_points
from env.base import ENV
import torch.nn as nn
import os
from pathlib import Path
from omegaconf import DictConfig
from utils.meckinova_utils import transform_trajectory_numpy
from utils.registry import Registry
from utils.colors import colors
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from tqdm import tqdm
import urchin
import meshcat
import time
import pybullet as p
from env.agent.mec_kinova import MecKinova
from env.base import create_enviroment
from env.sampler.mk_sampler import MecKinovaSampler
from env.scene.base_scene import Scene
from env.sim.bullet_simulator import BulletController
from eval.metrics import Evaluator
from models.dm.ddpm import DDPM
from models.model.unet import UNetModel
from models.optimizer.mk_motion_policy_optimization import MKMotionPolicyOptimizer
from models.planner.mkplanning import GreedyMKPlanner
from utils.misc import timestamp_str, compute_model_dim
from utils.io import dict2json, mkdir_if_not_exists
from datamodule.base import create_datamodule
from datamodule.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model
from typing import Dict, List, Optional, Sequence, Union
from utils.transform import SE3, transform_pointcloud_numpy
import pytorch_lightning as pl
import open3d as o3d
from cprint import cprint

@ENV.register()
class MKMotionPolicyRealEnv():
    def __init__(self, cfg: DictConfig):
        ## create evaluator and simulator
        if cfg.eval:
            self.eval = Evaluator(gui=cfg.sim_gui)

        else:
            self.eval = None
        ## create visualizer
        self.viz = meshcat.Visualizer() if cfg.viz else None
        self._viz_frame = cfg.viz_frame
        self._viz_time = cfg.viz_time
        self._init_viz()
        ## whether to save result
        self.save_dir = cfg.save_dir if cfg.save else None
        ## agent sampler
        self.mk_sampler = MecKinovaSampler('cpu', num_fixed_points=1024, use_cache=True)
    
    def _init_viz(self):
        if self.viz is not None:
            ## load the MK model
            self.agent_urdf = urchin.URDF.load(str(MecKinova.urdf_path))
            ## preload the robot meshes in meshcat at a neutral position
            for idx, (k, v) in enumerate(self.agent_urdf.visual_trimesh_fk(np.zeros(MecKinova.DOF)).items()):
                self.viz[f"robot/{idx}"].set_object(
                    meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
                    meshcat.geometry.MeshLambertMaterial(color=0x4F4F4F, wireframe=False),
                )
                self.viz[f"robot/{idx}"].set_transform(v)
    
    def _visualize_pointcloud(self, pc_name: str, pc_points: np.ndarray, pc_colors: np.ndarray, pc_size: float=0.015):
        if self.viz is not None:
            self.viz[pc_name].set_object(
                meshcat.geometry.PointCloud(
                    position=pc_points.T,
                    color=pc_colors.T,
                    size=pc_size,
                )
            )
    
    def _visualize_mesh(self, m_name, m):
        if m.visual.vertex_colors is not None:
            vertex_colors = np.asarray(m.visual.vertex_colors)[:, :3] / 255
            self.viz[m_name].set_object(
                meshcat.geometry.TriangularMeshGeometry(m.vertices, m.faces, vertex_colors),
                meshcat.geometry.MeshBasicMaterial(vertexColors=True),
            )
        else:
            self.viz[m_name].set_object(meshcat.geometry.TriangularMeshGeometry(m.vertices, m.faces))
    
    def _transform_pointcloud(self, pc_name: str, T: np.ndarray):
        if self.viz is not None:
            T = T.astype(np.float64)
            self.viz[pc_name].set_transform(T)
    
    def _transform_mesh(self, m_name: str, T: np.ndarray):
        if self.viz is not None:
            T = T.astype(np.float64)
            self.viz[m_name].set_transform(T)
    
    def _create_new_group(self, key: str):
        """
        Creates a new metric group (for a new setting, for example)
        :param key str: The key for this metric group
        """
        if self.eval:
            self.eval.current_result = {} # current trajectory evaluation result
            if key not in self.eval.groups:
                self.eval.groups[key] = {}
            self.eval.current_group_key = key
            self.eval.current_group = self.eval.groups[key]
    
    def print_overall_metrics(self):
        self.eval.print_overall_metrics()
    
    def evaluate(self, id: int, dt: float, time: float, data: Dict[str, torch.Tensor],
                    traj: Sequence[Union[Sequence, np.ndarray]], 
                    agent_object: object=MecKinova, skip_metrics: bool=False
    ):
        """ Evaluate the generated quality of the trajectory in the world frame
        """
        B = data['x'].shape[0]
        assert B == 1, 'the evaluation mode supports only 1 batch size'
        assert traj.ndim == 2, 'the trajectory of the evaluation must be 2 dimension'
        if self.eval:
            scene_name = data['scene_name'][0]
            task_name = data['task_name'][0]
            T_aw = data['T_aw'].squeeze(0).clone().detach().cpu().numpy()
            if task_name != 'goal-reach':
                object_name = data['object_name'][0].split('_')[0]
                self._create_new_group(f'{task_name}_{object_name}')
            else:
                self._create_new_group(f'{task_name}')

            ## convert agent trajectory to the world frame
            traj_a = copy.deepcopy(traj) # important
            initial_rot_angle = data['agent_init_pos'].clone().detach().squeeze(0).cpu().numpy()[-1]
            traj_w = transform_trajectory_numpy(traj_a, T_aw, initial_rot_angle)

            # NOTE: only the visualization of wireframes is supported
            # NOTE: we do not recommend using an sim gui because it is very slow to visualize the point of collision
            # NOTE: evaluation must be in agent initial frame, because joint limit is defined in agent initial frame 
            result = self.eval.evaluate_trajectory(
                dt=dt, time=time, trajectory=traj_a,
                obstacles_path=Scene(scene_name)._urdf_visual_path,
                obstacles_pose=SE3(np.linalg.inv(T_aw).astype(np.float64)),
                agent_object=agent_object, skip_metrics=skip_metrics,
            )

            ## save current trajectory evaluation result
            if self.save_dir is not None:
                mkdir_if_not_exists(os.path.join(self.save_dir, 'object'), True) # single object result
                mkdir_if_not_exists(os.path.join(self.save_dir, 'group'), True) # group statistical result
                mkdir_if_not_exists(os.path.join(self.save_dir, 'all'), True) # all statistical result
                self.save_result(id, traj_w, result)

            if task_name != 'goal-reach':
                cprint.info(f'Metrics for {task_name}ing {object_name}')
            else:
                cprint.info(f'Metrics for {task_name}ing task')
            self.eval.print_group_metrics()

    def save_result(self, id: int, trajectory_w: np.ndarray, eval_result: Dict):
        """ Save trajectory and evaluation result
        """
        item = eval_result
        item['trajectory_w'] = trajectory_w
        # save path
        object_save_path = os.path.join(self.save_dir, 'object', str(id) + '.json')
        group_save_path = os.path.join(self.save_dir, 'group', self.eval.current_group_key + '.json')
        all_save_path = os.path.join(self.save_dir, 'all', 'all.json')
        # save results
        dict2json(object_save_path, item)
        dict2json(group_save_path, self.eval.current_group)
        dict2json(all_save_path, self.eval.groups)
    
    def visualize(self, data: Dict[str, torch.Tensor], traj: Sequence[Union[Sequence, np.ndarray]]):
        """ Visualize Mesh and trajectory in webpage using Meshcat
        """
        assert traj.ndim == 2, 'the trajectory of the visualization must be 2 dimension'
        task_name = data['task_name'][len(data['task_name']) - 1]
        if 'object_name' in data.keys():
            object_name = data['object_name'][len(data['object_name']) - 1]

        if self.viz is not None:
            scene_name = data['scene_name'][len(data['scene_name']) - 1]
            scene = Scene(scene_name)
            scene_mesh = scene.get_link_in_scene('background_link')

            ## process transformation matrix
            T_aw = data['T_aw'].squeeze(0).clone().detach().cpu().numpy()
            T_wa = np.linalg.inv(T_aw).astype(np.float64)
            if task_name == 'pick':
                T_oa = data['T_oa'].squeeze(0).clone().detach().cpu().numpy()
                T_ow = np.matmul(T_aw, T_oa)
            elif task_name == 'place':
                T_oa_init = data['T_oa_init'].squeeze(0).clone().detach().cpu().numpy()
                T_ow_init = np.matmul(T_aw, T_oa_init)
                T_eeo = data['grasping_pose'].squeeze(0).clone().detach().cpu().numpy()
            ## agent initial rotation relative to world frame
            initial_rot_angle = data['agent_init_pos'].clone().detach().squeeze(0).cpu().numpy()[-1]

            if 'agent_pc_a' in data.keys():
                agent_pc_a = data['agent_pc_a'].squeeze(0).clone().detach().cpu().numpy()
                agent_pc_w = transform_pointcloud_numpy(agent_pc_a, T_aw)
                agent_pc_colors = np.expand_dims(np.array(colors[0]), 0).repeat(len(agent_pc_a), 0)
            if task_name == 'pick':
                object_mesh_w = scene.get_link(object_name).apply_transform(T_ow)
                object_mesh_a = scene.get_link(object_name).apply_transform(T_oa)
            elif task_name == 'place':
                object_mesh_o = scene.get_link(object_name)
            else: 
                target_pc_a = data['target_pc_a'].squeeze(0).clone().detach().cpu().numpy()
                target_pc_w = transform_pointcloud_numpy(target_pc_a, T_aw)
                target_pc_colors = np.expand_dims(np.array(colors[2]), 0).repeat(len(target_pc_a), 0)
            if task_name == 'place':
                placement_pc_a = data['scene_placement_pc_a'].squeeze(0).clone().detach().cpu().numpy()
                placement_pc_w = transform_pointcloud_numpy(placement_pc_a, T_aw)
                placement_pc_colors = np.expand_dims(np.array(colors[1]), 0).repeat(len(placement_pc_a), 0)

            ## visualize all point cloud
            if self._viz_frame == 'world_frame':
                # preprocess trajectory
                traj = transform_trajectory_numpy(traj, T_aw, initial_rot_angle)
                # visualize complete scene point cloud
                self._visualize_mesh('complete_scene', scene_mesh)
                # visualize agent initial point cloud
                if 'agent_pc_a' in data.keys(): self._visualize_pointcloud('agent_pc_w', agent_pc_w, agent_pc_colors, 0.050)
                # visualize object point cloud
                if task_name == 'pick':
                    self._visualize_mesh('object_mesh_w', object_mesh_w)
                elif task_name == 'place':
                    self._visualize_mesh('object_mesh_o', object_mesh_o)
                else:
                    self._visualize_pointcloud('target_pc_w', target_pc_w, target_pc_colors, 0.020)
                if task_name == 'place':
                    self._visualize_pointcloud('placement_pc_w', placement_pc_w, placement_pc_colors, 0.050)
            elif self._viz_frame == 'agent_initial_frame':
                # visualize agent initial point cloud
                if 'agent_pc_a' in data.keys(): self._visualize_pointcloud('agent_pc_a', agent_pc_a, agent_pc_colors, 0.050)
                # visualize object point cloud
                if task_name == 'pick':
                    self._visualize_mesh('object_mesh_a', object_mesh_a)
                elif task_name == 'place':
                    self._visualize_mesh('object_mesh_o', object_mesh_o)
                else:
                    self._visualize_pointcloud('target_pc_a', target_pc_a, target_pc_colors, 0.020)
                if task_name == 'place':
                    self._visualize_pointcloud('placement_pc_a', placement_pc_a, placement_pc_colors, 0.050)
                ## TODO
                ## visualize scene placement point cloud
                ## visualize target gripper point cloud
            
            #! tea pick: traj[-2] = traj[-1]
            traj[-2] = traj[-1]
            ## visualize agent motion
            for _ in range(self._viz_time):
                for cfg in traj:
                    # transform object point cloud, only for placement task
                    if task_name == 'place':
                        T_ee_cur = self.mk_sampler.end_effector_pose(torch.as_tensor(cfg)) \
                                    .squeeze(0).clone().detach().cpu().numpy()
                        T_o_cur = np.matmul(T_ee_cur, np.linalg.inv(T_eeo)) # T_ow or T_oa
                        self._transform_mesh('object_mesh_o', T_o_cur)
                    # transform agent link
                    for idx, (k, v) in enumerate(
                        self.agent_urdf.visual_trimesh_fk(cfg).items()
                    ):
                        self.viz[f"robot/{idx}"].set_transform(v)
                    time.sleep(0.1)
                time.sleep(0.2)



def scene_mesh_export(
    scene_name: str,
    object_name: str,
    T_aw: Union[List, np.ndarray],
    T_ow: Union[List, np.ndarray],
) -> str:
    """ Generate the mesh of the scene.
    """
    T_wa = np.linalg.inv(T_aw)
    scene = Scene(scene_name)
    scene.update_object_position_by_transformation_matrix(object_name, T_ow)
    # export mesh for pybullet
    if os.path.exists(str(Path(__file__).resolve().parent / "mesh")):
        shutil.rmtree(str(Path(__file__).resolve().parent / "mesh"))
    os.makedirs(str(Path(__file__).resolve().parent / "mesh"), exist_ok=True)
    scene_mesh_path = str(Path(__file__).resolve().parent / "mesh" / "main.obj")
    scene.transform(T_wa)
    scene.translation([0, 0, -0.02]) # filter collision between agent and scene flooring
    scene.export_collision(scene_mesh_path)
    return scene_mesh_path