import torch.nn as nn
from omegaconf import DictConfig
from utils.registry import Registry

ENV = Registry('Env')
def create_enviroment(cfg: DictConfig) -> nn.Module:
    """ Create a planning environment for planning task
    Args:
        cfg: configuration object
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A Plan Env
    """
    return ENV.get(cfg.name)(cfg)