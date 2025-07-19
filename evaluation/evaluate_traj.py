import os
import time
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from trimesh import transform_points
from env.agent.mec_kinova import MecKinova
from env.base import create_enviroment
from utils.meckinova_utils import transform_trajectory_torch
from utils.misc import compute_model_dim
from datamodule.base import create_datamodule

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def run_inference(config: DictConfig) -> None:
    if os.environ.get('SLURM') is not None:
        config.slurm = True # update slurm config
    
    device = f'cuda:0' if config.gpus is not None else 'cpu'

    ## prepare test dataset for evaluating on planning task
    dm = create_datamodule(cfg=config.task.datamodule, slurm=config.slurm)
    dl = dm.get_test_dataloader()

    ## create meckinova motion policy test environment
    env = create_enviroment(config.task.environment)

    ## evaluate
    with torch.no_grad():
        for i, data in enumerate(dl):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            start_time = time.time()

            ## TODO: 加载对应id为i的轨迹，轨迹需转换到机器人初始坐标系下（只需要转换x，y和theta，joint不用变）
            ## traj_unorm_a 数据类型为 np.ndarray

            ## evaluate trajectory
            env.evaluate(
                id=i,
                dt=0.08,  # we assume the time step for the trajectory is 0.08
                time=time.time() - start_time,
                data=data, traj=traj_unorm_a, agent_object=MecKinova
            )

            # visualize trajectory
            env.visualize(data, traj_unorm_a)
        print("Overall Metrics")
        env.print_overall_metrics()


if __name__ == '__main__':
    ## set random seed
    seed = 2024
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_inference()
