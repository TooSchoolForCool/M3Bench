from pathlib import Path
from typing import Dict
from cprint import *
import numpy as np
import torch
from datamodule.dataset.base import DATASET, DatasetType, MKPointCloudStateBase
from datamodule.dataset.transforms import make_default_transform

@DATASET.register()
class MKPointCloudInstanceDataset(MKPointCloudStateBase):
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
        return self.num_trajectories * self.expert_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        '''
        Returns a training datapoint representing a single configuration in a single scene 
        with the configuration at the next timestep as supervision

        Arguements:
            idx {int} -- Index represents the timestep within the trajectory
            Dict[str, torch.Tensor] -- The data used for training
        '''
        trajectory_idx, timestep = divmod(idx, self.expert_length)
        if timestep >= self.expert_length:
            timestep = self.expert_length - 1
        item = self.get_inputs(trajectory_idx, timestep)

        # Re-use the last point in the trajectory at the end
        supervision_timestep = np.clip(
            timestep + 1,
            0,
            self.expert_length - 1,
        )
        data = np.load(
            str(self._dir / str(trajectory_idx)) + '.npy', allow_pickle=True
        ).item() # load .npy and convert np.ndarray to List

        item["supervision"] = np.array(data['trajectory']['traj_a'][supervision_timestep])
        
        if self.transform is not None:
            item = self.transform(item)
        return item