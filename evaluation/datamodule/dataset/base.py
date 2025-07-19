from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset
from env.scene.base_scene import Scene
from preprocessing.data_utils import compute_scene_sdf
from utils.registry import Registry
import enum
import copy
from pathlib import Path
from utils.colors import colors
from typing import Dict
from cprint import *
import enum
import os
from torch.utils.data import Dataset
import numpy as np
import torch
from env.agent.mec_kinova import MecKinova
from env.sampler.mk_sampler import MecKinovaSampler
from utils.path import RootPath
from utils.pointcloud_utils import downsample_pointcloud
from trimesh import transform_points

DATASET = Registry('dataset')

class DatasetType(enum.Enum):
    """
    A simple enum class to indicate whether a dataloader is for training, validating, or testing.
    """
    TRAIN = 0
    VAL = 1
    TEST = 2

def create_dataset(cfg: dict, data_dir: Path, dataset_type: DatasetType, **kwargs: Dict) -> Dataset:
    """ Create a `torch.utils.data.Dataset` object from configuration.

    Args:
        cfg: configuration object, dataset configuration
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A Dataset object that has loaded the designated dataset.
    """
    if dataset_type == DatasetType.TRAIN:
        return DATASET.get(cfg.train_data_type)(cfg, data_dir, dataset_type, **kwargs)
    elif dataset_type == DatasetType.VAL:
        return DATASET.get(cfg.val_data_type)(cfg, data_dir, dataset_type, **kwargs)
    elif dataset_type == DatasetType.TEST:
        return DATASET.get(cfg.test_data_type)(cfg, data_dir, dataset_type, **kwargs)
    else:
        raise Exception(f"Invalid dataset type: {dataset_type}")

class MKPointCloudStateBase(Dataset):
    ''' State base dataset for meckinova motion policy
    '''

    def __init__(
        self,
        cfg: dict, 
        data_dir: Path, 
        dataset_type: DatasetType, 
        **kwargs: Dict,
    ):
        '''
        Arguments:
            directory {Path} -- The path to the root of the data directory
            trajectory_key {str} -- Generation strategy of expert, e.g. VKC
            num_agent_points {int} -- The number of points to sample from the agent
            num_scene_points {int} -- The number of points to sample from the scene
            num_object_points {int} -- The number of points to sample from the object object
            dataset_type {DatasetType} -- What type of dataset this is
            random_scale {float} -- The standard deviation of the random normal noise to apply 
                                    to the joints during training. This is only used for train datasets.
        '''
        self._init_directory(data_dir, dataset_type)
        self.trajectory_key = cfg.trajectory_key
        self.task_type = cfg.task_type
        self.train = dataset_type == DatasetType.TRAIN
        self.num_scene_points = cfg.num_scene_points
        self.num_agent_points = cfg.num_agent_points
        self.num_object_points = cfg.num_object_points
        self.num_placement_area_points = cfg.num_placement_area_points
        self.num_target_points = cfg.num_target_points
        self.random_scale = cfg.random_scale

        item = np.load(self._database, allow_pickle=True).item()
        self.expert_length = item['trajectory']['length'] # the length of a single trajectory
        self.agent_name = item['agent']['name'] # gets agent's name

        if self.agent_name == 'Mec_kinova':
            self.mk_sampler = MecKinovaSampler('cpu', num_fixed_points=self.num_agent_points, use_cache=True)
        else:
            raise Exception(f'Invalid agent name: {self.agent_name}')
    
    def _init_directory(self, directory: Path, dataset_type: DatasetType):
        '''
        Sets the path for the internal data structure based on the dataset type.

        Arguments:
            directory {Path} -- The path to the root of the data directory
            dataset_type {DatasetType} -- What type of dataset this is      
        Raises:
            Exception -- Raises an exception when the dataset type is unsupported
        '''
        self.type = dataset_type
        if dataset_type == DatasetType.TRAIN:
            self._dir = directory / 'train'
        elif dataset_type == DatasetType.VAL:
            self._dir = directory / 'val'
        elif dataset_type == DatasetType.TEST:
            self._dir = directory / 'test'
        else:
            raise Exception(f'Invalid dataset type: {dataset_type}')

        datalists = os.listdir(self._dir)
        datalists.sort()
        assert len(datalists) != 0, 'The data set cannot be empty.'
        self._num_trajectories = len(datalists) # The number of complete trajectories
        self._database = os.path.join(self._dir, datalists[0])
    
    @property
    def num_trajectories(self):
        '''
        Returns the total number of trajectories in the dataset
        '''
        return self._num_trajectories
    
    def get_inputs(self, trajectory_idx: int, timestep: int) -> Dict[str, torch.Tensor]:
        '''
        Loads all the relevant data and puts it in a dictionary. This includes normalizing all 
        configurations and constructing the pointcloud. If a training dataset, applies some 
        randomness to joints (before sampling the pointcloud).

        Arguments:
            trajectory_idx {int} -- The index of the trajectory in the train file
            timestep {int} -- The timestep within that trajectory
        Returns:
            Dict[str, torch.Tensor] -- The data used aggregated by the dataloader and used for training
        ------------------------------------------------------------------------------------------------
        item = {
            'scene_name': <scene name>,
            'object_name': <object name>,
            'agent_name': <agent name>,
            'task_name': <task name, 'pick', 'place' or 'goal-reach'>,
            'agent_init_pos': <agent initial position in world frame>,
            'traj_len': <the length of trajectory>,
            'grasping_pose': <grasping pose, [4, 4] matrix for object frame to gripper frame>,
            'T_aw': <[4, 4] matrix from world frame to agent frame>,
            'T_ow': <[4, 4] matrix from world frame to object frame>,
            'T_oa': <[4, 4] matrix from agent frame to object frame>,
            'scene_pc_a': <local scene point cloud in the agent frame>,
            'object_pc_a': <complete object point cloud in the agent frame>,
            'agent_pc_a': <complete agent point cloud at timestep in the agent frame>,
            'x': <mpidiffuser input, normalized trajectory in the agent frame>,
            'cfg': <mpinets and decision-transformer input, normalized configuration at timesetp in the agent frame>,
            'start': <mpidiffuser input, normalized configuration at inital time in the agent frame>,
            'end': <normalized configuration at final time in the agent frame>,
            'trans_mat': <[4, 4] matrix from agent frame to scene center frame>,
            'rot_angle': <The angle degree of the random rotation>,
            'pos': <mpidiffuser input, observed point cloud included scene, object, agent, placement area.etc.>,
            'xyz': <mpinets and decision-transformer input, observed point cloud and labels included scene, object, agent, placement area.etc.>,
            'feat': <mpidiffuser input, different parts of the point cloud have different colors>,
        }
        '''
        item = {} # Data used for training or validation
        data = np.load(str(self._dir / str(trajectory_idx)) + '.npy', allow_pickle=True).item() # load .npy and convert np.ndarray to List
        item['scene_name'] = data['scene']['name']
        item['agent_name'] = data['agent']['name']
        item['agent_init_pos'] = np.array(data['agent']['init_pos'])  
        item['traj_len'] = np.array(data['trajectory']['length'])
        item['task_name'] = self.task_type
        item['traj_a'] = np.array(data['trajectory']['traj_a'])
        item['traj_w'] = np.array(data['trajectory']['traj_w'])
        if self.task_type != 'goal-reach':
            item['object_name'] = data['object']['name']
        item['T_aw'] = np.array(data['transformation_matrix']['T_aw'])
        if self.task_type == 'pick':
            item['grasping_pose'] = np.array(data['task']['grasping_pose'])
            item['T_ow'] = np.array(data['transformation_matrix']['T_ow'])
            item['T_oa'] = np.array(data['transformation_matrix']['T_oa'])
        elif self.task_type == 'place':
            item['supporter'] = data['task']['supporter']
            item['grasping_pose'] = np.array(data['task']['grasping_pose'])
            item['T_ow_final'] = np.array(data['transformation_matrix']['T_ow_final'])
            item['T_oa_init'] = np.array(data['transformation_matrix']['T_oa_init'])

        #! 点云的合并放在 tranform 中，点云要降采样，点云和颜色需要打乱
        #! 如果是 goal-reach 的话，给的是 gripper 的点云
        item['scene_pc_a'] = downsample_pointcloud(
            pc=np.array(data['scene']['pointcloud']['points']),
            sample_num=self.num_scene_points,
            shuffle=True
        ) # local scene point cloud in the agent frame
        if self.task_type == 'pick':
            item['object_pc_a'] = downsample_pointcloud(
                pc=np.array(data['object']['pointcloud']['points']),
                sample_num=self.num_object_points,
                shuffle=True
            ) # complete inital object point cloud in the agent frame
        elif self.task_type == 'place':
            if self.train:
                item['scene_placement_pc_a'] = downsample_pointcloud(
                    pc=np.array(data['task']['placement_area']['scene_placement_pc']['points']),
                    sample_num=self.num_placement_area_points,
                    shuffle=True
                ) # scene placement surface point cloud in agent frame
            else:
                item['scene_placement_pc_a'] = downsample_pointcloud(
                    pc=np.array(data['task']['placement_area']['scene_placement_pc']['points_test']),
                    sample_num=self.num_placement_area_points,
                    shuffle=True
                ) # scene placement surface point cloud in agent frame
            # object placement undersurface point cloud in self frame
            item['object_placement_pc_o'] = np.array(data['task']['placement_area']['object_placement_pc']['points'])

        # ensure that after adding random noise, the joint angles are still within the joint limits
        limits = MecKinova.JOINT_LIMITS
        traj_a = np.minimum(
            np.maximum(np.array(data['trajectory']['traj_a']), limits[:, 0]), limits[:, 1]
        )
        item['x'] = traj_a # trajectory in the agent frame

        # for goal-reaching task, acquire the point cloud of the gripper given the taregt pose
        if self.task_type == 'goal-reach':
            item['target_pc_a'] = self.mk_sampler.sample_end_effector(
                torch.as_tensor(traj_a[-1]).float(), self.num_target_points, True
            ).squeeze(0).clone().detach().cpu().numpy()

        cfg = copy.deepcopy(traj_a[timestep]) # current configuration in the timestep

        # for placement task, the point cloud of the object is dynamic
        if self.task_type == 'place':
            # when agent's current configuration is cfg
            T_eea = self.mk_sampler.end_effector_pose(
                torch.as_tensor(cfg).float()
            ).squeeze(0).clone().detach().cpu().numpy()
            T_oa = np.matmul(T_eea, np.linalg.inv(item['grasping_pose']))
            object_pc_o = transform_points(
                np.array(data['object']['pointcloud']['points']), 
                np.linalg.inv(item['T_oa_init'])
            )
            item['object_pc_a'] = downsample_pointcloud(
                pc=transform_points(object_pc_o, T_oa),
                sample_num=self.num_object_points,
                shuffle=True
            ) # complete current object point cloud in the agent frame
            item['object_placement_pc_a'] = downsample_pointcloud(
                pc=transform_points(item['object_placement_pc_o'], T_oa),
                sample_num=self.num_object_points,
                shuffle=True
            ) # object placement point cloud in the agent frame

        if self.train:
            # add slight random noise to the joints
            randomized = self.random_scale * np.random.randn(cfg.shape[0]) + cfg
            # clamp to joint limits
            randomized = np.minimum(np.maximum(randomized, limits[:, 0]), limits[:, 1])
            item['configuration'] = randomized # normalize configuration  
            item['agent_pc_a'] = self.mk_sampler.sample(
                torch.as_tensor(randomized).float()
            ).squeeze(0).clone().detach().cpu().numpy()
        else:
            item['configuration'] = cfg
            item['agent_pc_a'] = self.mk_sampler.sample(
                torch.as_tensor(cfg).float()
            ).squeeze(0).clone().detach().cpu().numpy()
        
        item['trans_mat'] = np.eye(4)
        item['rot_angle'] = 0

        # # debug & visualize
        # import open3d as o3d
        # pcd1 = o3d.geometry.PointCloud()
        # pcd2 = o3d.geometry.PointCloud()
        # pcd3 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(item['agent_pc_a'])
        # pcd2.points = o3d.utility.Vector3dVector(item['scene_pc_a'])
        # pcd3.points = o3d.utility.Vector3dVector(item['object_pc_a'])
        # pcd1.paint_uniform_color([0, 0, 1.0])
        # pcd2.paint_uniform_color([0, 1.0, 0])
        # pcd3.paint_uniform_color([1.0, 0, 0])
        # o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

        if self.task_type == 'pick':
            ## item['pos'] and item['feat'] are input for mpidiffuser and decision-transformer
            item['pos'] = np.concatenate((item['scene_pc_a'], item['object_pc_a']), axis=0)
            item['feat'] = np.concatenate(
                (
                    np.expand_dims(np.array(colors[1]), 0).repeat(self.num_scene_points, 0), 
                    np.expand_dims(np.array(colors[2]), 0).repeat(self.num_object_points, 0), #! 如果往下面加点云，点云的编号需要增加
                ), axis=0,
            )
            
            ## item['cfg'] and item['xyz'] are input for mpinets
            item['xyz'] = np.concatenate(
                (
                    np.zeros((self.num_agent_points, 4)), # the agent point cloud is labeled 0
                    np.ones((self.num_scene_points, 4)), # the scene point cloud is labeled 1
                    2 * np.ones((self.num_object_points, 4)), # the object point cloud is labeled 2
                ), axis=0,
            )
            item['xyz'][:self.num_agent_points, :3] = item['agent_pc_a']
            item['xyz'][self.num_agent_points:self.num_agent_points + self.num_scene_points, :3] = item['scene_pc_a']
            item["xyz"][self.num_agent_points + self.num_scene_points:, :3] = item['object_pc_a']
        elif self.task_type == 'place':
            item['pos'] = np.concatenate((item['scene_pc_a'], item['object_pc_a'], item['scene_placement_pc_a']), axis=0)
            item['feat'] = np.concatenate(
                (
                    np.expand_dims(np.array(colors[1]), 0).repeat(self.num_scene_points, 0), 
                    np.expand_dims(np.array(colors[2]), 0).repeat(self.num_object_points, 0),
                    np.expand_dims(np.array(colors[3]), 0).repeat(self.num_placement_area_points, 0),
                ), axis=0,
            )
            ## item['cfg'] and item['xyz'] are input for mpinets
            item['xyz'] = np.concatenate(
                (
                    np.zeros((self.num_agent_points, 4)), # the agent point cloud is labeled 0
                    np.ones((self.num_scene_points, 4)), # the scene point cloud is labeled 1
                    2 * np.ones((self.num_object_points, 4)), # the object point cloud is labeled 2
                    3 * np.ones((self.num_placement_area_points, 4)), # the scene placement point cloud is labeled 3
                ), axis=0,
            )
            item['xyz'][:self.num_agent_points, :3] = item['agent_pc_a']
            item['xyz'][self.num_agent_points:self.num_agent_points + self.num_scene_points, :3] = item['scene_pc_a']
            item["xyz"][self.num_agent_points + self.num_scene_points:-self.num_placement_area_points, :3] = item['object_pc_a']
            item["xyz"][-self.num_placement_area_points:, :3] = item['scene_placement_pc_a']
        elif self.task_type == 'goal-reach':
            ## item['pos'] and item['feat'] are input for mpidiffuser and decision-transformer
            item['pos'] = np.concatenate((item['scene_pc_a'], item['target_pc_a']), axis=0)
            item['feat'] = np.concatenate(
                (
                    np.expand_dims(np.array(colors[1]), 0).repeat(self.num_scene_points, 0), 
                    np.expand_dims(np.array(colors[2]), 0).repeat(self.num_target_points, 0),
                ), axis=0,
            )
            
            ## item['cfg'] and item['xyz'] are input for mpinets
            item['xyz'] = np.concatenate(
                (
                    np.zeros((self.num_agent_points, 4)), # the agent point cloud is labeled 0
                    np.ones((self.num_scene_points, 4)), # the scene point cloud is labeled 1
                    2 * np.ones((self.num_target_points, 4)), # the target point cloud is labeled 2
                ), axis=0,
            )
            item['xyz'][:self.num_agent_points, :3] = item['agent_pc_a']
            item['xyz'][self.num_agent_points:self.num_agent_points + self.num_scene_points, :3] = item['scene_pc_a']
            item["xyz"][self.num_agent_points + self.num_scene_points:, :3] = item['target_pc_a']
        else: 
            raise Exception(f"Invalid dataset type: {self.task_type}")

        ## load SDF Data
        sdf_path = str(RootPath.SCENE / item['scene_name'] / 'sdf.npy')
        if not os.path.exists(sdf_path):
            sdf_dict = compute_scene_sdf(scene=Scene(item['scene_name']), size=128)
            np.save(sdf_path, sdf_dict)
        sdf_dict = np.load(sdf_path, allow_pickle=True).item()
        item['sdf_norm_value'] = np.array(sdf_dict['sdf_norm_value'])
        item['scene_mesh_center'] = np.array(sdf_dict['scene_mesh_center'])
        item['scene_mesh_scale'] = np.array(sdf_dict['scene_mesh_scale'])
        item['resolution'] = np.array(sdf_dict['resolution'])

        return item


class MKPointCloudSquenceBase(Dataset):
    ''' Squence base dataset for meckinova motion policy
    '''

    def __init__(
        self,
        cfg: dict, 
        data_dir: Path, 
        dataset_type: DatasetType, 
        **kwargs: Dict,
    ):
        '''
        Arguments:
            directory {Path} -- The path to the root of the data directory
            trajectory_key {str} -- Generation strategy of expert, e.g. VKC
            num_agent_points {int} -- The number of points to sample from the agent
            num_scene_points {int} -- The number of points to sample from the scene
            num_object_points {int} -- The number of points to sample from the object object
            dataset_type {DatasetType} -- What type of dataset this is
            random_scale {float} -- The standard deviation of the random normal noise to apply 
                                    to the joints during training. This is only used for train datasets.
        '''
        self._init_directory(data_dir, dataset_type)
        self.trajectory_key = cfg.trajectory_key
        self.context_length = cfg.context_length
        self.model_embed_timesteps = cfg.embed_timesteps
        self.max_predicted_length = cfg.max_predicted_length
        assert self.model_embed_timesteps >= self.max_predicted_length, \
            'model embedding timesteps should be greater than or equal to max predicted length of trajectory'
        self.task_type = cfg.task_type
        self.train = dataset_type == DatasetType.TRAIN
        self.num_scene_points = cfg.num_scene_points
        self.num_agent_points = cfg.num_agent_points
        self.num_object_points = cfg.num_object_points
        self.num_placement_area_points = cfg.num_placement_area_points
        self.num_target_points = cfg.num_target_points
        self.random_scale = cfg.random_scale

        item = np.load(self._database, allow_pickle=True).item()
        self.expert_length = self.max_predicted_length - self.context_length # NOTE: It's not the true length of the trajectory
        self.agent_name = item['agent']['name'] # gets agent's name

        if self.agent_name == 'Mec_kinova':
            self.mk_sampler = MecKinovaSampler('cpu', num_fixed_points=self.num_agent_points, use_cache=True)
        else:
            raise Exception(f'Invalid agent name: {self.agent_name}')
    
    def _init_directory(self, directory: Path, dataset_type: DatasetType):
        '''
        Sets the path for the internal data structure based on the dataset type.

        Arguments:
            directory {Path} -- The path to the root of the data directory
            dataset_type {DatasetType} -- What type of dataset this is      
        Raises:
            Exception -- Raises an exception when the dataset type is unsupported
        '''
        self.type = dataset_type
        if dataset_type == DatasetType.TRAIN:
            self._dir = directory / 'train'
        elif dataset_type == DatasetType.VAL:
            self._dir = directory / 'val'
        elif dataset_type == DatasetType.TEST:
            self._dir = directory / 'test'
        else:
            raise Exception(f'Invalid dataset type: {dataset_type}')

        datalists = os.listdir(self._dir)
        datalists.sort()
        assert len(datalists) != 0, 'The data set cannot be empty.'
        self._num_trajectories = len(datalists) # The number of complete trajectories
        self._database = os.path.join(self._dir, datalists[0])
    
    @property
    def num_trajectories(self):
        '''
        Returns the total number of trajectories in the dataset
        '''
        return self._num_trajectories
    
    def get_inputs(self, trajectory_idx: int, timestep: int) -> Dict[str, torch.Tensor]:
        '''
        Loads all the relevant data and puts it in a dictionary. This includes normalizing all 
        configurations and constructing the pointcloud. If a training dataset, applies some 
        randomness to joints (before sampling the pointcloud).

        Arguments:
            trajectory_idx {int} -- The index of the trajectory in the train file
            timestep {int} -- The timestep within that trajectory
        Returns:
            Dict[str, torch.Tensor] -- The data used aggregated by the dataloader and used for training
        ------------------------------------------------------------------------------------------------
        item = {
            'scene_name': <scene name>,
            'object_name': <object name>,
            'agent_name': <agent name>,
            'task_name': <task name, 'pick', 'place' or 'goal-reach'>,
            'agent_init_pos': <agent initial position in world frame>,
            'traj_len': <the length of trajectory>,
            'grasping_pose': <grasping pose, [4, 4] matrix for object frame to gripper frame>,
            'T_aw': <[4, 4] matrix from world frame to agent frame>,
            'T_ow': <[4, 4] matrix from world frame to object frame>,
            'T_oa': <[4, 4] matrix from agent frame to object frame>,
            'scene_pc_a': <local scene point cloud in the agent frame>,
            'object_pc_a': <complete object point cloud in the agent frame>,
            'agent_pc_a': <complete agent point cloud at timestep in the agent frame>,
            'x': <mpidiffuser input, normalized trajectory in the agent frame>,
            'cfg': <mpinets and decision-transformer input, normalized configuration at timesetp in the agent frame>,
            'start': <mpidiffuser input, normalized configuration at inital time in the agent frame>,
            'end': <normalized configuration at final time in the agent frame>,
            'trans_mat': <[4, 4] matrix from agent frame to scene center frame>,
            'rot_angle': <The angle degree of the random rotation>,
            'pos': <mpidiffuser input, observed point cloud included scene, object, agent, placement area.etc.>,
            'xyz': <mpinets and decision-transformer input, observed point cloud and labels included scene, object, agent, placement area.etc.>,
            'feat': <mpidiffuser input, different parts of the point cloud have different colors>,
        }
        '''
        item = {} # Data used for training or validation
        data = np.load(str(self._dir / str(trajectory_idx)) + '.npy', allow_pickle=True).item() # load .npy and convert np.ndarray to List
        item['scene_name'] = data['scene']['name']
        item['agent_name'] = data['agent']['name']
        item['agent_init_pos'] = np.array(data['agent']['init_pos'])  
        item['traj_len'] = np.array(data['trajectory']['length'])
        item['traj_a'] = np.array(data['trajectory']['traj_a'])
        item['traj_w'] = np.array(data['trajectory']['traj_w'])
        item['task_name'] = self.task_type
        if self.task_type != 'goal-reach':
            item['object_name'] = data['object']['name']
        #! task 如果是放置的话，还需要添加别的数据
        item['T_aw'] = np.array(data['transformation_matrix']['T_aw'])
        if self.task_type == 'pick':
            item['grasping_pose'] = np.array(data['task']['grasping_pose']) 
            item['T_ow'] = np.array(data['transformation_matrix']['T_ow'])
            item['T_oa'] = np.array(data['transformation_matrix']['T_oa'])
        elif self.task_type == 'place':
            item['grasping_pose'] = np.array(data['task']['grasping_pose']) 
            item['supporter'] = data['task']['supporter']
            item['T_ow_final'] = np.array(data['transformation_matrix']['T_ow_final'])
            item['T_oa_init'] = np.array(data['transformation_matrix']['T_oa_init'])

        #! 如果是 goal-reach 的话，给的是 gripper 的点云
        item['scene_pc_a'] = downsample_pointcloud(
            pc=np.array(data['scene']['pointcloud']['points']),
            sample_num=self.num_scene_points,
            shuffle=True
        ) # local scene point cloud in the agent frame
        scene_pc_a_sq = np.stack([item['scene_pc_a'] for _ in range(self.context_length)], axis=0)
        if self.task_type == 'pick':
            item['object_pc_a'] = downsample_pointcloud(
                pc=np.array(data['object']['pointcloud']['points']),
                sample_num=self.num_object_points,
                shuffle=True
            ) # complete object point cloud in the agent frame
            object_pc_a_sq = np.stack([item['object_pc_a'] for _ in range(self.context_length)], axis=0)
        elif self.task_type == 'place':
            item['object_pc_a'] = downsample_pointcloud(
                pc=np.array(data['object']['pointcloud']['points']),
                sample_num=self.num_object_points,
                shuffle=True
            ) # complete inital object point cloud in the agent frame
            item['scene_placement_pc_a'] = downsample_pointcloud(
                pc=np.array(data['task']['placement_area']['scene_placement_pc']['points']),
                sample_num=self.num_placement_area_points,
                shuffle=True
            ) # scene placement surface point cloud in agent frame
            scene_placement_pc_a_sq = np.stack([item['scene_placement_pc_a'] for _ in range(self.context_length)], axis=0)
            # object placement undersurface point cloud in self frame
            item['object_placement_pc_o'] = np.array(data['task']['placement_area']['object_placement_pc']['points'])

        # ensure that after adding random noise, the joint angles are still within the joint limits
        limits = MecKinova.JOINT_LIMITS
        traj_a = np.minimum(
            np.maximum(np.array(data['trajectory']['traj_a']), limits[:, 0]), limits[:, 1]
        )
        item['x'] = traj_a # trajectory in the agent frame

        # for goal-reaching task, acquire the point cloud of the gripper given the taregt pose
        if self.task_type == 'goal-reach':
            item['target_pc_a'] = self.mk_sampler.sample_end_effector(
                torch.as_tensor(traj_a[-1]).float(), self.num_target_points, True
            ).squeeze(0).clone().detach().cpu().numpy()
            target_pc_a_sq = np.stack([item['target_pc_a'] for _ in range(self.context_length)], axis=0)

        # according to max predicted length, extend original trajectory
        traj_a_extension = copy.deepcopy(traj_a)
        for _ in range(self.max_predicted_length - traj_a.shape[0]): 
            traj_a_extension = np.concatenate((traj_a_extension, traj_a[-1:]), axis=0)

        # current configuration squence in the timestep
        cfg_sq = traj_a_extension[timestep: timestep + self.context_length]

        # for placement task, the point cloud of the object is dynamic
        if self.task_type == 'place':
            # when agent's current configuration is cfg
            T_eea_sq = self.mk_sampler.end_effector_pose(
                torch.as_tensor(cfg_sq).float()
            ).clone().detach().cpu().numpy()
            T_oa_sq = np.einsum('ijk, kl->ijl', T_eea_sq, np.linalg.inv(item['grasping_pose']))
            object_pc_o = transform_points(
                np.array(data['object']['pointcloud']['points']), 
                np.linalg.inv(item['T_oa_init'])
            )
            object_pc_a_list = []
            for i in range(self.context_length):
                object_pc_a_cur = downsample_pointcloud(
                    pc=transform_points(copy.deepcopy(object_pc_o), T_oa_sq[i]),
                    sample_num=self.num_object_points,
                    shuffle=True
                ) # complete current object point cloud in the agent frame
                object_pc_a_list.append(object_pc_a_cur)
            object_pc_a_sq = np.stack(object_pc_a_list, axis=0)

        if self.train:
            # add slight random noise to the joints
            randomized = self.random_scale * np.random.randn(cfg_sq.shape[0], cfg_sq.shape[1]) + cfg_sq
            # clamp to joint limits
            randomized = np.minimum(np.maximum(randomized, limits[:, 0]), limits[:, 1])
            item['configuration_sq'] = randomized # normalize configuration  
            agent_pc_a_sq = self.mk_sampler.sample(
                torch.as_tensor(randomized).float()
            ).squeeze(0).clone().detach().cpu().numpy() # [context_length, N, 3]
        else:
            item['configuration_sq'] = cfg_sq
            agent_pc_a_sq = self.mk_sampler.sample(
                torch.as_tensor(cfg_sq).float()
            ).squeeze(0).clone().detach().cpu().numpy()
        
        ## transformation matrix and rotation angle
        item['trans_mat'] = np.eye(4)
        item['rot_angle'] = 0

        ## process point cloud data
        if self.task_type == 'pick':
            item['pos'] = np.concatenate((item['scene_pc_a'], item['object_pc_a']), axis=0)
            ## item['cfg_sq'] and item['xyz_sq'] are input for mpitransformer
            xyz = np.concatenate(
                (
                    np.zeros((self.num_agent_points, 4)), # the agent point cloud is labeled 0
                    np.ones((self.num_scene_points, 4)), # the scene point cloud is labeled 1
                    2 * np.ones((self.num_object_points, 4)), # the object point cloud is labeled 2
                ), axis=0,
            )
            item['xyz_sq'] = np.stack([xyz for _ in range(self.context_length)], axis=0)
            item['xyz_sq'][:, :self.num_agent_points, :3] = agent_pc_a_sq
            item['xyz_sq'][:, self.num_agent_points:self.num_agent_points + self.num_scene_points, :3] = scene_pc_a_sq
            item['xyz_sq'][:, self.num_agent_points + self.num_scene_points:, :3] = object_pc_a_sq
        elif self.task_type == 'place':
            ## item['cfg_sq'] and item['xyz_sq'] are input for mpitransformer
            xyz = np.concatenate(
                (
                    np.zeros((self.num_agent_points, 4)), # the agent point cloud is labeled 0
                    np.ones((self.num_scene_points, 4)), # the scene point cloud is labeled 1
                    2 * np.ones((self.num_object_points, 4)), # the object point cloud is labeled 2
                    3 * np.ones((self.num_placement_area_points, 4)), # the scene placement point cloud is labeled 3
                ), axis=0,
            )
            item['xyz_sq'] = np.stack([xyz for _ in range(self.context_length)], axis=0)
            item['xyz_sq'][:, :self.num_agent_points, :3] = agent_pc_a_sq
            item['xyz_sq'][:, self.num_agent_points:self.num_agent_points + self.num_scene_points, :3] = scene_pc_a_sq
            item['xyz_sq'][:, self.num_agent_points + self.num_scene_points:-self.num_placement_area_points, :3] = object_pc_a_sq
            item['xyz_sq'][:, -self.num_placement_area_points:, :3] = scene_placement_pc_a_sq
        elif self.task_type == 'goal-reach':
            item['pos'] = np.concatenate((item['scene_pc_a'], item['target_pc_a']), axis=0)
            ## item['cfg_sq'] and item['xyz_sq'] are input for mpitransformer
            xyz = np.concatenate(
                (
                    np.zeros((self.num_agent_points, 4)), # the agent point cloud is labeled 0
                    np.ones((self.num_scene_points, 4)), # the scene point cloud is labeled 1
                    2 * np.ones((self.num_target_points, 4)), # the target point cloud is labeled 2
                ), axis=0,
            )
            item['xyz_sq'] = np.stack([xyz for _ in range(self.context_length)], axis=0)
            item['xyz_sq'][:, :self.num_agent_points, :3] = agent_pc_a_sq
            item['xyz_sq'][:, self.num_agent_points:self.num_agent_points + self.num_scene_points, :3] = scene_pc_a_sq
            item['xyz_sq'][:, self.num_agent_points + self.num_scene_points:, :3] = target_pc_a_sq
        else: 
            raise Exception(f"Invalid dataset type: {self.task_type}")

        ## load SDF Data
        sdf_path = str(RootPath.SCENE / item['scene_name'] / 'sdf.npy')
        if not os.path.exists(sdf_path):
            sdf_dict = compute_scene_sdf(scene=Scene(item['scene_name']), size=128)
            np.save(sdf_path, sdf_dict)
        sdf_dict = np.load(sdf_path, allow_pickle=True).item()
        item['sdf_norm_value'] = np.array(sdf_dict['sdf_norm_value'])
        item['scene_mesh_center'] = np.array(sdf_dict['scene_mesh_center'])
        item['scene_mesh_scale'] = np.array(sdf_dict['scene_mesh_scale'])
        item['resolution'] = np.array(sdf_dict['resolution'])

        ## squence timesteps and attention masks
        item['timesteps'] = np.arange(timestep, timestep + self.context_length)
        item['attention_masks'] = np.ones(self.context_length)

        ## supervision squence
        item['supervision_sq'] = traj_a_extension[timestep + 1: timestep + self.context_length + 1]
        item['context_length'] = self.context_length
        item['max_predicted_length'] = self.max_predicted_length

        return item