import string
import random
from datetime import datetime
from omegaconf import DictConfig

from env.agent.mec_kinova import MecKinova

def timestamp_str() -> str:
    """ Get current time stamp string
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")

def random_str(length: int=4) -> str:
    """ Generate random string with given length
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=4))

def compute_model_dim(cfg: DictConfig) -> int:
    """ Compute modeling dimension for different task

    Args:
        cfg: task configuration
    
    Return:
        The modeling dimension
    """
    if cfg.agent == 'MecKinova':
        return MecKinova.DOF
    else:
        raise Exception('Unsupported task.')

