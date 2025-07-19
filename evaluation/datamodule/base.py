from typing import Dict
from utils.registry import Registry
import pytorch_lightning as pl

DATAMODULE = Registry('datamodule')

def create_datamodule(cfg: dict, slurm: bool, **kwargs: Dict) -> pl.LightningDataModule:
    """ Create a `torch.utils.data.Dataset` object from configuration.

    Args:
        cfg: configuration object, dataset configuration
        slurm: on slurm platform or not. This field is used to specify the data path
    
    Return:
        A Dataset object that has loaded the designated dataset.
    """
    return DATAMODULE.get(cfg.name)(cfg, slurm, **kwargs)
