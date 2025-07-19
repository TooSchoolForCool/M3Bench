import itertools
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
from typing import Any, Dict, List, Optional, Sequence
import trimesh
import open3d as o3d
import trimesh.creation
from trimesh import transform_points
import trimesh.sample
from natsort import ns, natsorted
import theseus as th
import gc

sys.path.append("../../")
from env.scene.base_scene import Scene
from env.agent.mec_kinova import MecKinova
from utils.io import dict2json, mkdir_if_not_exists
from utils.transform import SE3
from third_party.grasp_diffusion.se3dif.utils.geometry_utils import SO3_R3
from env.sampler.mk_sampler import MecKinovaSampler
from third_party.grasp_diffusion.se3dif.visualization import grasp_visualization
from third_party.grasp_diffusion.se3dif.models.loader import load_model
from third_party.grasp_diffusion.se3dif.samplers.grasp_samplers import Grasp_AnnealedLD


""" 
python eval_all_result_pick_dataset.py --result_dir ../../results_dataset/pick/${timestamp} --dataset_test_dir /your_data_path/pick/test
"""

# they can be gripped
SKIP_OBJ_NAME = {
    'kettle',
    'bread',
    'tomato',
}

TEST_SET_LIST = [
    'seen-seen',
    'seen-unseen',
    'unseen-seen',
    'unseen-unseen'
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--robot', type=str, default='MecKinova', help='robot name, such as MecKinova or Franka')
    p.add_argument('--task', type=str, default='pick', help='task name, such as pick, or place')
    p.add_argument('--result_dir', type=str, help='inference result directory path')
    p.add_argument('--dataset_test_dir', type=str, help='dataset test directory')

    opt = p.parse_args()
    return opt

def percent_true(arr: Sequence) -> float:
    """
    Returns the percent true of a boolean sequence or the percent nonzero of a numerical sequence

    :param arr Sequence: The input sequence
    :rtype float: The percent
    """
    return 100 * np.count_nonzero(arr) / len(arr)

def add_metric(group, key, value):
    group[key] = group.get(key, []) + [value]

def eval_metrics(group: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates the metrics for a specific group

    :param group Dict[str, Any]: The group of results
    :rtype Dict[str, float]: The metrics
    """
    #! There was a problem with the previous code, so let's rework it here
    group["physical_violations"] = (
        group["collision"] 
        or group["joint_limit_violation"]
        or group["self_collision"]
    )
    group["physical_success"] = [not x for x in group["physical_violations"]]
    ## --------------------------------------------------------------------
    number = np.sum(group["number"])
    physical_success = percent_true(group["physical_success"])
    physical = percent_true(group["physical_violations"])
    config_smoothness = np.mean(group["config_smoothness"])
    eff_smoothness = np.mean(group["eff_smoothness"])
    all_eff_position_path_lengths = np.asarray(group["eff_position_path_length"])
    all_eff_orientation_path_lengths = np.asarray(
        group["eff_orientation_path_length"]
    )
    all_t_dist = np.asarray(group["t_dist"])
    all_r_dist = np.asarray(group["r_dist"])
    all_obj_z_dist = np.asarray(group["obj_z_dist"])
    all_obj_z_init = np.asarray(group["obj_z_init"])
    all_times = np.asarray(group["time"])

    is_smooth = percent_true(
        np.logical_and(
            np.asarray(group["config_smoothness"]) < -1.6,
            np.asarray(group["eff_smoothness"]) < -1.6,
        )
    )

    physical_successes = group["physical_success"]

    physical_success_position_path_lengths = all_eff_position_path_lengths[
        list(physical_successes)
    ]
    physical_success_orientation_path_lengths = all_eff_orientation_path_lengths[
        list(physical_successes)
    ]
    
    if len(physical_success_position_path_lengths) > 0:
        eff_position_path_length = (
            np.mean(physical_success_position_path_lengths),
            np.std(physical_success_position_path_lengths),
        )
    else:
        eff_position_path_length = (0, 0)

    task_success = np.logical_and(
        np.logical_and(
            all_t_dist < 0.02,
            all_obj_z_init > 0.5
        ),
        np.logical_and(
            all_r_dist < 15,
            all_obj_z_dist > 0.08
        )
    )
    # print(all_t_dist)

    is_success = percent_true(
        np.logical_and(physical_successes, task_success)
    )

    if len(physical_success_orientation_path_lengths) > 0:
        eff_orientation_path_length = (
            np.mean(physical_success_orientation_path_lengths),
            np.std(physical_success_orientation_path_lengths),
        )
    else:
        eff_orientation_path_length = (0, 0)    

    physical_success_times = all_times[list(group["physical_success"])]

    if len(physical_success_times) > 0:
        time = (
            np.mean(physical_success_times),
            np.std(physical_success_times),
        )
    else:    
        time = (0, 0)

    collision = percent_true(group["collision"])
    joint_limit = percent_true(group["joint_limit_violation"])
    self_collision = percent_true(group["self_collision"])
    depths = np.array(
        list(itertools.chain.from_iterable(group["collision_depths"]))
    )
    if len(depths) == 0: depths = [0]
    all_num_steps = np.asarray(group["num_steps"])
    physical_success_num_steps = all_num_steps[list(physical_successes)]
    if len(physical_success_num_steps) > 0:
        step_time = (
            np.mean(physical_success_times / physical_success_num_steps),
            np.std(physical_success_times / physical_success_num_steps),
        )
    else:
        step_time = (0, 0)

    return {
        "time": time,
        "number": number,
        "step time": step_time,
        "is success": is_success,
        "physical_success": physical_success,
        "env collision": collision,
        "self collision": self_collision,
        "joint violation": joint_limit,
        "physical violations": physical,
        "average collision depth": 100 * np.mean(depths),
        "median collision depth": 100 * np.median(depths),
        "is smooth": is_smooth,
        "average config sparc": config_smoothness,
        "average eff sparc": eff_smoothness,
        "eff position path length": eff_position_path_length,
        "eff orientation path length": eff_orientation_path_length,
    }

def print_metrics(group: Dict[str, Any]):
    metrics = eval_metrics(group)
    print(f"% Success: {metrics['is success']:4.2f}")
    print(f"% With Environment Collision: {metrics['env collision']:4.2f}")
    print(f"% With Self Collision: {metrics['self collision']:4.2f}")
    print(f"% With Joint Limit Violations: {metrics['joint violation']:4.2f}")
    print(f"Average Collision Depth (cm): {metrics['average collision depth']}")
    print(f"Median Collision Depth (cm): {metrics['median collision depth']}")
    print(f"Average Config SPARC: {metrics['average config sparc']:4.2f}")
    print(f"Average End Eff SPARC: {metrics['average eff sparc']:4.2f}")
    print(f"% Smooth: {metrics['is smooth']:4.2f}")
    print(
        "Average End Eff Position Path Length:"
        f" {metrics['eff position path length'][0]:4.2f}"
        f" ± {metrics['eff position path length'][1]:4.2f}"
    )
    print(
        "Average End Eff Orientation Path Length:"
        f" {metrics['eff orientation path length'][0]:4.2f}"
        f" ± {metrics['eff orientation path length'][1]:4.2f}"
    )
    print(f"Average Time: {metrics['time'][0]:4.2f} ± {metrics['time'][1]:4.2f}")
    print(
        "Average Time Per Step (Not Always Valuable):"
        f" {metrics['step time'][0]:4.6f}"
        f" ± {metrics['step time'][1]:4.6f}"
    )

def get_metrics(group: Dict[str, Any]):
    metrics = eval_metrics(group)
    return {
        "% Success": f"{metrics['is success']:4.2f}",
        "Number": f"{metrics['number']}",
        "% With Environment Collision": f"{metrics['env collision']:4.2f}",
        "% With Self Collision": f"{metrics['self collision']:4.2f}",
        "% With Joint Limit Violations": f"{metrics['joint violation']:4.2f}",
        "Average Collision Depth (cm)": f"{metrics['average collision depth']}",
        "Median Collision Depth (cm)": f"{metrics['median collision depth']}",
        "Average Config SPARC": f"{metrics['average config sparc']:4.2f}",
        "Average End Eff SPARC": f"{metrics['average eff sparc']:4.2f}",
        "% Smooth": f"{metrics['is smooth']:4.2f}",
        "Average End Eff Position Path Length": f"{metrics['eff position path length'][0]:4.2f} ± {metrics['eff position path length'][1]:4.2f}",
        "Average End Eff Orientation Path Length": f" {metrics['eff orientation path length'][0]:4.2f} ± {metrics['eff orientation path length'][1]:4.2f}",
        "Average Time": f"{metrics['time'][0]:4.2f} ± {metrics['time'][1]:4.2f}",
        "Average Time Per Step (Not Always Valuable)": f" {metrics['step time'][0]:4.6f} ± {metrics['step time'][1]:4.6f}"
    }

def save_metrics(save_path: str, eval_group: Dict[str, Any]):
    item = {}
    for key in eval_group.keys():
        item[key] = get_metrics(eval_group[key])
    dict2json(save_path, item)

def eval_pick_result(result_dir: str, test_dir: str):
    object_result_dir = os.path.join(result_dir, 'object')
    pick_eval_res_path = os.path.join(result_dir, 'eval_res.json')
    pick_test_info_path = os.path.join(os.path.dirname(test_dir),'test_info.json')
    with open(pick_eval_res_path, "r") as f:
        pick_eval_res = json.load(f) 
    with open(pick_test_info_path, "r") as f:
        pick_test_info = json.load(f)

    eval_group = {}
    # create evaluation group
    for test_set_state in TEST_SET_LIST:
        eval_group[test_set_state] = {}
        eval_group[test_set_state]["all"] = {}
    
    mk_agent = MecKinova()
    id_iter = tqdm(sorted(os.listdir(object_result_dir), key=lambda x:int(x.split('.')[0])), desc="{0: ^10}".format('ID'), leave=False, mininterval=0.1)
    for id_name in id_iter:
        id = id_name.split('.')[0]
        if id not in pick_eval_res.keys(): continue
        result_file_path = os.path.join(object_result_dir, id_name)
        with open(os.path.join(result_file_path), "r") as f:
            result_item = json.load(f)
        test_file_path = os.path.join(test_dir, id + '.npy')
        test_data = np.load(test_file_path, allow_pickle=True).item()
        object_name = test_data['object']['name'].split('_')[0]

        if object_name in SKIP_OBJ_NAME:
            continue

        test_set_state = pick_test_info[id]['state'][0] + '-' + pick_test_info[id]['state'][1]

        traj_w = result_item['trajectory_w']
        cfg_init = copy.deepcopy(traj_w[-1])
        cfg_final = copy.deepcopy(traj_w[-1]); cfg_final[4] = cfg_final[4] * 0.7
        T_eew_init_se3 = SE3(matrix=mk_agent.get_eff_pose(cfg_init))
        T_eeo_init_se3 = SE3(matrix=np.array(pick_eval_res[id]['H_init']))
        T_ow_init_se3 = T_eew_init_se3.__matmul__(T_eeo_init_se3.inverse)
        T_eew_final_se3 = SE3(matrix=mk_agent.get_eff_pose(cfg_final))
        T_eeo_final_se3 = SE3(matrix=np.array(pick_eval_res[id]['H_final']))
        T_ow_final_se3 = T_eew_final_se3.__matmul__(T_eeo_final_se3.inverse)
        obj_z_dist = T_ow_final_se3._xyz[-1] - T_ow_init_se3._xyz[-1]

        # if pick_eval_res[id]["t_dist(m)"] < 0.02 and pick_eval_res[id]["r_dist(degree)"] < 15 and \
        #     obj_z_dist > 0.05 and T_ow_init_se3._xyz[-1] > 0.5 and test_set_state == 'seen-unseen' \
        #     and object_name == 'spraybottle':
        #     print(id)
            
        
        # all results
        add_metric(eval_group[test_set_state]["all"], "collision_depths", result_item["collision_depths"])
        add_metric(eval_group[test_set_state]["all"], "collision", result_item["collision"])
        add_metric(eval_group[test_set_state]["all"], "physical_success", result_item["physical_success"])
        add_metric(eval_group[test_set_state]["all"], "physical_violations", result_item["physical_violations"])
        add_metric(eval_group[test_set_state]["all"], "joint_limit_violation", result_item["joint_limit_violation"])
        add_metric(eval_group[test_set_state]["all"], "self_collision", result_item["self_collision"])
        add_metric(eval_group[test_set_state]["all"], "config_smoothness", result_item["config_smoothness"])
        add_metric(eval_group[test_set_state]["all"], "eff_smoothness", result_item["eff_smoothness"])
        add_metric(eval_group[test_set_state]["all"], "joint_limit_violation", result_item["joint_limit_violation"])
        add_metric(eval_group[test_set_state]["all"], "eff_position_path_length", result_item["eff_position_path_length"])
        add_metric(eval_group[test_set_state]["all"], "eff_orientation_path_length", result_item["eff_orientation_path_length"])
        add_metric(eval_group[test_set_state]["all"], "time", result_item["time"])
        add_metric(eval_group[test_set_state]["all"], "num_steps", result_item["num_steps"])
        add_metric(eval_group[test_set_state]["all"], "t_dist", pick_eval_res[id]["t_dist(m)"])
        add_metric(eval_group[test_set_state]["all"], "r_dist", pick_eval_res[id]["r_dist(degree)"])
        add_metric(eval_group[test_set_state]["all"], "obj_z_dist", obj_z_dist)
        add_metric(eval_group[test_set_state]["all"], "obj_z_init", T_ow_init_se3._xyz[-1])
        add_metric(eval_group[test_set_state]["all"] , "number", 1)

        # object results
        if object_name not in eval_group[test_set_state].keys():
            eval_group[test_set_state][object_name] = {}
        add_metric(eval_group[test_set_state][object_name] , "collision_depths", result_item["collision_depths"])
        add_metric(eval_group[test_set_state][object_name] , "collision", result_item["collision"])
        add_metric(eval_group[test_set_state][object_name] , "physical_success", result_item["physical_success"])
        add_metric(eval_group[test_set_state][object_name] , "physical_violations", result_item["physical_violations"])
        add_metric(eval_group[test_set_state][object_name] , "joint_limit_violation", result_item["joint_limit_violation"])
        add_metric(eval_group[test_set_state][object_name] , "self_collision", result_item["self_collision"])
        add_metric(eval_group[test_set_state][object_name] , "config_smoothness", result_item["config_smoothness"])
        add_metric(eval_group[test_set_state][object_name] , "eff_smoothness", result_item["eff_smoothness"])
        add_metric(eval_group[test_set_state][object_name] , "joint_limit_violation", result_item["joint_limit_violation"])
        add_metric(eval_group[test_set_state][object_name] , "eff_position_path_length", result_item["eff_position_path_length"])
        add_metric(eval_group[test_set_state][object_name] , "eff_orientation_path_length", result_item["eff_orientation_path_length"])
        add_metric(eval_group[test_set_state][object_name] , "time", result_item["time"])
        add_metric(eval_group[test_set_state][object_name] , "num_steps", result_item["num_steps"])
        add_metric(eval_group[test_set_state][object_name] , "t_dist", pick_eval_res[id]["t_dist(m)"])
        add_metric(eval_group[test_set_state][object_name] , "r_dist", pick_eval_res[id]["r_dist(degree)"])
        add_metric(eval_group[test_set_state][object_name] , "obj_z_dist", obj_z_dist)
        add_metric(eval_group[test_set_state][object_name] , "obj_z_init", T_ow_init_se3._xyz[-1])
        add_metric(eval_group[test_set_state][object_name] , "number", 1)
        
    for test_set_state in TEST_SET_LIST:
        if len(eval_group[test_set_state]['all']) == 0:
            continue
        save_metrics(os.path.join(result_dir, 'eval_metrics_' + test_set_state + '.json'), eval_group[test_set_state])


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

    eval_pick_result(args.result_dir, args.dataset_test_dir)
