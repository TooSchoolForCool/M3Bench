import copy
import torch
import numpy as np
from trimesh import transform_points
from utils.transform import transform_pointcloud_torch

def transform_trajectory_torch(
    trajs: torch.Tensor,
    trans_mats: torch.Tensor,
    rot_angles: torch.Tensor,
) -> torch.Tensor:
    """
    Arguements:
        trajectories {torch.Tensor} -- Unnormalized agent trajectory, [B, L, agent_dof].
        trans_mats {torch.Tensor} -- Transformation matrix, [B, 4, 4].
        rot_angles {torch.Tensor} -- Rotation angle in z axis, [B, 1].
    Returns:
        torch.Tensor -- The transformed agent trajectory, [B, L, agent_dof].
    """
    trajectories = trajs
    xy_2ds = trajectories[:,:,:2] # [B, L, 2]
    xy_3ds = torch.concatenate(
        (
            xy_2ds, torch.zeros(
                (xy_2ds.shape[0], xy_2ds.shape[1], 1)
            ).to(device=trajectories.device)
        ), 
        dim=-1,
    ) # [B, L, 3]
    xy_3ds = transform_pointcloud_torch(xy_3ds, trans_mats)
    trajectories[:,:,:2] = xy_3ds[:,:,:2]
    trajectories[:,:,2] = trajectories[:,:,2] + rot_angles.unsqueeze(1)
    return trajectories

def transform_configuration_torch(
    cfgs: torch.Tensor,
    trans_mats: torch.Tensor,
    rot_angles: torch.Tensor,
) -> torch.Tensor:
    """
    Arguements:
        configurations {torch.Tensor} -- Unnormalized agent configuration, [B, agent_dof].
        trans_mats {torch.Tensor} -- Transformation matrix, [B, 4, 4].
        rot_angles {torch.Tensor} -- Rotation angle in z axis, [B, 1].
    Returns:
        torch.Tensor -- The transformed agent configuration, [B, agent_dof].
    """
    B = cfgs.shape[0]
    configurations = cfgs
    xy_2d = configurations[:,:2] # [B, 2]
    xy_3d = torch.concatenate((xy_2d, torch.zeros((B, 1)).to(device=configurations.device)), dim=-1) # [B, 3]
    xy_3d = transform_pointcloud_torch(xy_3d.unsqueeze(1), trans_mats).squeeze(1)
    configurations[:,:2] = xy_3d[:,:2]
    configurations[:,2] = configurations[:,2] + rot_angles
    return configurations

def transform_trajectory_numpy(
    traj: np.ndarray,
    trans_mat: np.ndarray,
    rot_angle: np.ndarray,
) -> np.ndarray:
    """
    Arguements:
        trajectory {np.ndarray} -- Unnormalized agent trajectory, [L, agent_dof].
        trans_mat {np.ndarray} -- Transformation matrix, [4, 4].
        rot_angle {np.ndarray} -- Rotation angle in z axis, [1].
    Returns:
        torch.Tensor -- The transformed agent trajectory, [L, agent_dof].
    """
    trajectory = copy.deepcopy(traj)
    xy_2ds = trajectory[:,:2]
    xy_3ds = np.concatenate((xy_2ds, np.zeros((xy_2ds.shape[0], 1))), axis=-1)
    xy_3ds = transform_points(xy_3ds, trans_mat).astype(np.float64)
    trajectory[:,:2] = xy_3ds[:,:2]
    trajectory[:,2] = trajectory[:,2] + rot_angle
    return trajectory

def transform_configuration_numpy(
    cfg: np.ndarray,
    trans_mat: np.ndarray,
    rot_angle: np.ndarray,
) -> np.ndarray:
    """
    Arguements:
        configuration {np.ndarray} -- Unnormalized agent configuration, [agent_dof].
        trans_mat {np.ndarray} -- Transformation matrix, [4, 4].
        rot_angle {np.ndarray} -- Rotation angle in z axis, [1].
    Returns:
        np.ndarray -- The transformed agent configuration, [agent_dof].
    """
    configuration = copy.deepcopy(cfg)
    xy_2d = configuration[:2]
    xy_3d = np.concatenate((xy_2d, np.zeros(1)), axis=-1)
    xy_3d = np.squeeze(transform_points(np.expand_dims(xy_3d, axis=0), trans_mat).astype(np.float64))
    configuration[:2] = xy_3d[:2]
    configuration[2] = configuration[2] + rot_angle
    return configuration