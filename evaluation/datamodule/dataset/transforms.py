from typing import Any, Dict, List
import torch
import numpy as np
from trimesh import transform_points
from datamodule.dataset.base import DatasetType
from env.agent.mec_kinova import MecKinova
from utils.meckinova_utils import transform_configuration_numpy, transform_trajectory_numpy


class Compose(object):
    """ Composes several transforms together
    """

    def __init__(self, transforms: Any) -> None:
        self.transforms = transforms

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Any: 
        for t in self.transforms:
            args = t(data, *args, **kwargs)
        return args


class NumpyToTensor(object):
    """ Convert numpy data to torch.Tensor data
    """
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        tensor_float_list = [
            'agent_init_pos', 'traj_len', 'grasping_pose', 'T_aw', 'T_ow', 'T_oa', 'scene_pc_a',
            'object_pc_a', 'agent_pc_a', 'x', 'configuration', 'supervision', 'start', 'end', 
            'trans_mat', 'rot_angle', 'pos', 'xyz', 'feat', 'Hs', 'sdf_norm_value', 'scene_mesh_center',
            'scene_mesh_scale', 'resolution', 'configuration_sq', 'supervision_sq', 'xyz_sq', 
            'crop_center', 'T_ow_final', 'T_oa_init', 'scene_placement_pc_a', 'object_placement_pc_o',
            'scene_placement_pc_a_sq', 'target_pc_a', 'object_placement_pc_a', 'object_placement_pc_a_sq',
            'traj_a', 'traj_w'
        ]
        tensor_long_list = ['timesteps', 'attention_masks', 'context_length', 'max_predicted_length']
        for key in data.keys():
            if key in tensor_float_list and not torch.is_tensor(data[key]):
                data[key] = torch.as_tensor(np.array(data[key])).float()
            elif key in tensor_long_list and not torch.is_tensor(data[key]):
                data[key] = torch.as_tensor(np.array(data[key])).long()
        return data


class RandomRotationZ(object):
    """ Random rotation z augmentation
    """
    def __init__(self, **kwargs) -> None:
        self.angle = [0, 0, 1]

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> dict:
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi

        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=np.float64)
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float64)
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=np.float64)
        trans_mat = np.eye(4, dtype=np.float64)
        trans_mat[0:3, 0:3] = np.dot(R_z, np.dot(R_y, R_x))
        # transform point cloud
        if 'pos' in data.keys(): data['pos'] = transform_points(data['pos'], trans_mat).astype(np.float64)
        if 'xyz' in data.keys(): data['xyz'][:,:-1] = transform_points(data['xyz'][:,:-1], trans_mat).astype(np.float64)
        if 'xyz_sq' in data.keys(): 
            for i in range(data['context_length']):
                data['xyz_sq'][i,:,:-1] = transform_points(data['xyz_sq'][i,:,:-1], trans_mat).astype(np.float64)
        # transform agent trajectory
        if 'x' in data.keys(): data['x'] = transform_trajectory_numpy(data['x'], trans_mat, angle_z)
        # transform agent configuration at timestep
        if 'configuration' in data.keys(): data['configuration'] = transform_configuration_numpy(data['configuration'], trans_mat, angle_z)
        if 'configuration_sq' in data.keys():
            for i in range(data['context_length']):
                data['configuration_sq'][i] = transform_configuration_numpy(data['configuration_sq'][i], trans_mat, angle_z).astype(np.float64)
        # transform agent supervision at timestep
        if 'supervision' in data.keys(): data['supervision'] = transform_configuration_numpy(data['supervision'], trans_mat, angle_z)
        if 'supervision_sq' in data.keys():
            for i in range(data['context_length']):
                data['supervision_sq'][i] = transform_configuration_numpy(data['supervision_sq'][i], trans_mat, angle_z).astype(np.float64)
        
        data['trans_mat'] = trans_mat @ data['trans_mat']
        data['rot_angle'] = angle_z

        return data


class NormalizeToCenterPath(object):
    """ Normalize scene to center
    """
    def __init__(self, **kwargs) -> None:
        self.gravity_dim = kwargs['gravity_dim'] 

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Dict:
        xyz = data['pos']
        # center = data['crop_center']
        center = (xyz.max(axis=0) + xyz.min(axis=0)) * 0.5
        center[self.gravity_dim] = np.percentile(xyz[:, self.gravity_dim], 1)
        trans_mat = np.eye(4, dtype=np.float64)
        trans_mat[0:3, -1] -= center
        # transform point cloud
        if 'pos' in data.keys(): data['pos'] = transform_points(data['pos'], trans_mat).astype(np.float64)
        if 'xyz' in data.keys(): data['xyz'][:,:-1] = transform_points(data['xyz'][:,:-1], trans_mat).astype(np.float64)
        if 'xyz_sq' in data.keys(): 
            for i in range(data['context_length']):
                data['xyz_sq'][i,:,:-1] = transform_points(data['xyz_sq'][i,:,:-1], trans_mat).astype(np.float64)
        # transform agent trajectory
        if 'x' in data.keys(): data['x'] = transform_trajectory_numpy(data['x'], trans_mat, 0)
        # transform agent configuration at timestep
        if 'configuration' in data.keys(): data['configuration'] = transform_configuration_numpy(data['configuration'], trans_mat, 0)
        if 'configuration_sq' in data.keys():
            for i in range(data['context_length']):
                data['configuration_sq'][i] = transform_configuration_numpy(data['configuration_sq'][i], trans_mat, 0).astype(np.float64)
        # transform agent supervision at timestep
        if 'supervision' in data.keys(): data['supervision'] = transform_configuration_numpy(data['supervision'], trans_mat, 0)
        if 'supervision_sq' in data.keys():
            for i in range(data['context_length']):
                data['supervision_sq'][i] = transform_configuration_numpy(data['supervision_sq'][i], trans_mat, 0).astype(np.float64)
        
        data['trans_mat'] = trans_mat @ data['trans_mat']

        return data


class NormalizePolicyData(object):
    """ Normalize the configuration and trajectory data
    """
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, data: Dict, *args: List, **kwargs: Dict) -> Dict:
        if 'x' in data.keys():
            data['x'] = MecKinova.normalize_joints(data['x'])
            data['start'] = np.expand_dims(data['x'][0], axis=0)
        if 'configuration' in data.keys(): data['configuration'] = MecKinova.normalize_joints(data['configuration'])
        if 'supervision' in data.keys(): data['supervision'] = MecKinova.normalize_joints(data['supervision'])
        if 'configuration_sq' in data.keys(): data['configuration_sq'] = MecKinova.normalize_joints(data['configuration_sq'])
        if 'supervision_sq' in data.keys(): data['supervision_sq'] = MecKinova.normalize_joints(data['supervision_sq'])
        
        # # debug
        # import open3d as o3d
        # pcd2 = o3d.geometry.PointCloud()
        # pcd3 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(data['pos'][0: 0 + 4096, :3])
        # pcd3.points = o3d.utility.Vector3dVector(data['pos'][0 + 4096:, :3])
        # pcd2.paint_uniform_color([0, 1.0, 0])
        # pcd3.paint_uniform_color([1.0, 0, 0])
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
        # o3d.visualization.draw_geometries([frame, pcd2, pcd3])
        # exit()

        return data


TRANSFORMS = {
    'NumpyToTensor': NumpyToTensor,
    'RandomRotationZ': RandomRotationZ,
    'NormalizeToCenterPath': NormalizeToCenterPath,
    'NormalizePolicyData': NormalizePolicyData,
}


def make_default_transform(cfg: dict, datatype: DatasetType) -> Compose:
    """ Make default transform

    Args:
        cfg: global configuration
        phase: process phase
    
    Return:
        Composed transforms.
    """
    ## generate transform configuration
    transform_cfg = {'phase': datatype, **cfg.transform_cfg}

    ## compose
    transforms = []
    if datatype == DatasetType.TRAIN:
        transforms_list = cfg.train_transforms
    elif datatype == DatasetType.VAL:
        transforms_list = cfg.val_transforms
    elif datatype == DatasetType.TEST:
        transforms_list = cfg.test_transforms

    for t in transforms_list:
        transforms.append(TRANSFORMS[t](**transform_cfg))

    return Compose(transforms)