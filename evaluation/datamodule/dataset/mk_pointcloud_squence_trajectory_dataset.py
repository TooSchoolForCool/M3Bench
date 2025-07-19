from pathlib import Path
from typing import Dict
from cprint import *
import torch
from datamodule.dataset.base import DATASET, DatasetType, MKPointCloudSquenceBase
from datamodule.dataset.transforms import make_default_transform

@DATASET.register()
class MKPointCloudSquenceTrajectoryDataset(MKPointCloudSquenceBase):
    '''
    This is the dataset used primarily for training. Each element in the dataset represents the agent and 
    scene at a particular time {t}. Likewise, the supervision is the agent's configuration at q_{t+1}.
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
            random_scale {float} -- The standard deviation of the random normal noise to apply 
                                    to the joints during training. This is only used for train datasets.
        '''
        super().__init__(cfg, data_dir, dataset_type, **kwargs)
        self.transform = make_default_transform(cfg, dataset_type)
    
    def __len__(self):
        '''
        Returns the total number of start configurations in the dataset (i.e. the length of 
        the trajectories times the number of trajectories)
        '''
        return self.num_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        '''
        Returns a training datapoint representing a single configuration in a single scene 
        with the configuration at the next timestep as supervision

        Arguements:
            idx {int} -- Index represents the timestep within the trajectory
            Dict[str, torch.Tensor] -- The data used for training
        '''
        trajectory_idx, timestep = idx, 0
        item = self.get_inputs(trajectory_idx, timestep)
        if self.transform is not None:
            item = self.transform(item)
        return item