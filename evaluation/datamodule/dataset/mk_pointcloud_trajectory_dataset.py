from pathlib import Path
from typing import Dict
from cprint import *
import numpy as np
import torch
from datamodule.dataset.base import DATASET, DatasetType, MKPointCloudStateBase
from datamodule.dataset.transforms import make_default_transform

@DATASET.register()
class MKPointCloudTrajectoryDataset(MKPointCloudStateBase):
    '''
    This dataset is used exclusively for validating. Each element in the dataset represents a trajectory start and 
    scene. There is no supervision because this is used to produce an entire rollout and check for success. When 
    doing validation, we care more about success than we care about matching the expert's behavior (which is a key 
    difference from training).
    '''
    def __init__(
        self,
        cfg: dict, 
        data_dir: Path, 
        dataset_type: DatasetType, 
        **kwargs: Dict,
    ):
        '''
        Arguements:
            directory {Path} -- The path to the root of the data directory
            num_agent_points {int} -- The number of points to sample from the agent
            num_scene_points {int} -- The number of points to sample from the scene
            num_object_points {int} -- The number of points to sample from the object
            dataset_type {DatasetType} -- What type of dataset this is
        '''
        super().__init__(cfg, data_dir, dataset_type, **kwargs)
        self.transform = make_default_transform(cfg, dataset_type)

    def __len__(self):
        '''
        Necessary for Pytorch. For this dataset, the length is the total number of problems.
        '''
        return self.num_trajectories
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        '''
        Required by Pytorch. Queries for data at a particular index. Note that in this dataset, 
        the index always corresponds to the trajectory index.

        Arguements:
            idx {int} -- The index
        Returns:
            Dict[str, torch.Tensor] -- Returns a dictionary that can be assembled by the data 
                                       loader before using in training.
        '''
        trajectory_idx, timestep = idx, 0
        item = self.get_inputs(trajectory_idx, timestep)
        if self.transform is not None:
            item = self.transform(item)
        return item