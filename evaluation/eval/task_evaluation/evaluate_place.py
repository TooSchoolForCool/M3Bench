import sys
import copy
import json
import os
import time
import argparse
import random
import sys
import numpy as np
from cprint import cprint
import torch
import math
from tqdm import tqdm
from typing import Dict, List, Optional
import trimesh
import open3d as o3d
import trimesh.creation
from trimesh import transform_points
import trimesh.sample
from natsort import ns, natsorted
import theseus as th

sys.path.append("../../")
from utils.transform import SE3
from third_party.grasp_diffusion.se3dif.utils.geometry_utils import SO3_R3
from env.sampler.mk_sampler import MecKinovaSampler
from third_party.grasp_diffusion.se3dif.visualization import grasp_visualization
from third_party.grasp_diffusion.se3dif.models.loader import load_model
from third_party.grasp_diffusion.se3dif.samplers.grasp_samplers import Grasp_AnnealedLD


""" python evaluate_pick.py --result_dir ../../results/mk_mpidiffuser_pick/2024-06-18-09-58-39 --dataset_test_dir ../../../mpidiffuser-data/distance_0_cm_data/data/pick/test
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--robot', type=str, default='MecKinova', help='robot name, such as MecKinova or Franka')
    p.add_argument('--task', type=str, default='pick', help='task name, such as pick, or place')
    p.add_argument('--result_dir', type=str, help='inference result directory path')
    p.add_argument('--dataset_test_dir', type=str, help='dataset test directory')
    p.add_argument('--sample_num', type=int, default=500, help='number of sampled grasping pose for evaluation')

    opt = p.parse_args()
    return opt

def process_pick_result(object_result_dir: str, dataset_test_dir: str, sample_num: int, sampler: MecKinovaSampler):
    id_iter = tqdm(sorted(os.listdir(object_result_dir)), desc="{0: ^10}".format('ID'), leave=False, mininterval=0.1)
    for id_name in id_iter:
        id = id_name.split('.')[0]
        if id != '11':
            continue
        result_file_path = os.path.join(object_result_dir, id_name)
        test_file_path = os.path.join(dataset_test_dir, id + '.npy')
        ## result file
        with open(os.path.join(result_file_path), "r") as f:
            result_item = json.load(f) 
        ## the corresponding test set file
        test_data = np.load(test_file_path, allow_pickle=True).item()

        ## generate good grasping pose for evaluating 'pick' result
        if 'object_grasping_poses' not in test_data.keys():
            device = 'cuda:0'
            object_pc_points = np.array(test_data['object']['pointcloud']['points'])
            T_oa = np.array(test_data['transformation_matrix']['T_oa'])
            T_ao = np.linalg.inv(T_oa)
            object_pc_o = trimesh.transform_points(object_pc_points, T_ao)
            # normalize object point clouds
            object_pc_o *= 8.
            object_pcs_o_mean = np.mean(object_pc_o, axis=0)
            object_pc_o -= object_pcs_o_mean
            object_pc_o = torch.as_tensor(object_pc_o).float().to(device=device)
            # load model
            args = argparse.Namespace(device=device, model='grasp_dif_multi', batch=sample_num)
            model_args = {'device': device, 'pretrained_model': args.model}
            grasp_energy_model = load_model(model_args)
            context = object_pc_o[None, ...]
            grasp_energy_model.set_latent(context, batch=args.batch)
            # sample grasping poses
            generator = Grasp_AnnealedLD(grasp_energy_model, batch=args.batch, T=70, T_fit=50, k_steps=2, device=args.device)
            Hs = generator.sample()
            # # visualize results an debug
            # object_pc_o *= 1 / 8
            # Hs[..., :3, -1] *= 1 / 8.
            # grasp_visualization.visualize_grasps(
            #     Hs=Hs.clone().detach().cpu().numpy(), 
            #     p_cloud=object_pc_o.clone().detach().cpu().numpy()
            # )
            # exit()
            Hs = Hs.clone().detach().cpu().numpy()
            Hs[..., :3, -1] = (Hs[..., :3, -1] + object_pcs_o_mean) / 8.
            test_data['object_grasping_poses'] = {
                'values': Hs,
                'description': 'object candidate grasping poses',
            }
            np.save(test_file_path, test_data)
            Hs = torch.as_tensor(Hs)
        else:
            Hs = torch.as_tensor(test_data['object_grasping_poses']['values'])
        
        ## evaluate grasping pose
        traj_w = np.array(result_item['trajectory_w'])
        object_pc_points = np.array(test_data['object']['pointcloud']['points'])
        T_oa = np.array(test_data['transformation_matrix']['T_oa'])
        T_ao = np.linalg.inv(T_oa)
        object_pc_o = trimesh.transform_points(object_pc_points, T_ao)
        T_ow = np.array(test_data['transformation_matrix']['T_ow'])
        T_wo = np.linalg.inv(T_ow)
        T_eew = sampler.end_effector_pose(torch.as_tensor(traj_w[-1])) # [1, 4, 4] (torch.Tensor)
        T_eeo = torch.matmul(torch.as_tensor(T_wo).float(), T_eew).repeat(sample_num, 1, 1) # [sample_num, 4, 4] (torch.Tensor)
        # compute translation distance
        t_dists = torch.norm((T_eeo[:,:3,-1] - Hs[:,:3,-1]), p=2, dim=-1) # [sample_num]
        # compute rotation distance
        Hs_lie = SO3_R3(R=Hs[:,:3,:3], t=Hs[:,:3, -1]) # [sample_num, 4, 4]
        T_oee = torch.inverse(T_eeo)
        T_oee_lie = SO3_R3(R=T_oee[:, :3, :3], t=T_oee[:, :3, -1]) # [sample_num, 4, 4]
        w = th.compose(T_oee_lie.R, Hs_lie.R); r = w.log_map()
        r_dists = torch.norm(r, p=2, dim=-1) # [sample_num]
        # compute SE(3) distance
        se3_dists = 5 * t_dists + r_dists
        _, min_index = torch.min(se3_dists, dim=0)
        # the closest good grasping pose
        Hs_closest = SE3(Hs[min_index.int(),...].clone().detach().cpu().numpy())
        T_eeo = SE3(T_eeo[min_index.int(),...].clone().detach().cpu().numpy())
        t_closest_dist = np.linalg.norm(T_eeo._xyz - Hs_closest._xyz)
        cprint.info(f'{T_eeo._xyz}, {Hs_closest._xyz}')
        r_closest_dist = np.abs(np.degrees((T_eeo.so3._quat * Hs_closest.so3._quat.conjugate).radians))
        cprint.info(f'{t_closest_dist}, {r_closest_dist}')
        grasp_visualization.visualize_grasps(
            Hs=np.stack([Hs_closest.matrix, T_eeo.matrix]),
            p_cloud=object_pc_o,
        )

        

if __name__ == "__main__":
    args = parse_args()

    ## set random seed
    seed = 2024
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    mk_sampler = MecKinovaSampler('cpu', num_fixed_points=1024, use_cache=True)
    process_pick_result(os.path.join(args.result_dir, 'object'), args.dataset_test_dir, args.sample_num, mk_sampler)
