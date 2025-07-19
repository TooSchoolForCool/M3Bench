from datetime import datetime
import json
import argparse
import shutil
from typing import Optional
import torch
import gc
import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from natsort import ns, natsorted
import tongverse as tv
from tongverse.env import Env
from tongverse.sensor import Camera, default_camera_cfg
# from geometrout.transform import SE3, SO3
from transform import SE3
from cprint import *
from copy import deepcopy
from tongverse.env import Env, default_env_cfg
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.materials.physics_material import PhysicsMaterial

#! NOTE: 如果要保存图片，请在命令行中添加 --save_image 参数，并执行以下操作
#! 进入isaac sim时迅速进入点击暂停，调整灯光，demo_light和stage_light

"""
python evaluate_place.py --result_dir /your_ws_path/m3bench/results/place/timestamp --dataset_test_dir /your_data_path/place/test 
"""

class NumpyArrayEncoder(json.JSONEncoder):
    """
    Python dictionaries can store ndarray array types, but when serialized by dict into JSON files, 
    ndarray types cannot be serialized. In order to read and write numpy arrays, 
    we need to rewrite JSONEncoder's default method. 
    The basic principle is to convert ndarray to list, and then read from list to ndarray.
    """
    def default(self, obj):
        return obj.tolist()

def dict2json(file_name: str, the_dict: dict) -> None:
    """
    Save dict object as json.
    """
    try:
        json_str = json.dumps(the_dict, cls=NumpyArrayEncoder, indent=4)
        with open(file_name, 'w') as json_file:
            json_file.write(json_str)
        return 1
    except:
        cprint.err('something wrong')
        return 0 
    
def rmdir_if_exists(dir_name: str, recursive: bool=False) -> None:
    """ Remove directory with the given dir_name
    Args:
        dir_name: input directory name that can be a path
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
        
def mkdir_if_not_exists(dir_name: str, recursive: bool=False) -> None:
    """ Make directory with the given dir_name
    Args:
        dir_name: input directory name that can be a path
        recursive: recursive directory creation
    """
    if os.path.exists(dir_name):
        return 
    
    if recursive:
        os.makedirs(dir_name)
    else:
        os.mkdir(dir_name)

def timestamp_str() -> str:
    """ Get current time stamp string
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--robot', type=str, default='MecKinova', help='robot name, such as MecKinova or Franka')
    p.add_argument('--task', type=str, default='pick', help='task name, such as pick, or place')
    p.add_argument('--result_dir', type=str, help='inference result directory path')
    p.add_argument('--dataset_test_dir', type=str, help='dataset test directory')
    p.add_argument('--save_image', action="store_true", help='whether to save the images of the detection process')

    opt = p.parse_args()
    return opt

START = 100
END = 600
LIGHT_ADJUSTMENT = False

def calculate_iou(rect1, rect2, union_type:Optional[str]=None):
    """
    Calculate the Intersection over Union (IOU) of two rectangles.

    Parameters:
    rect1, rect2: Each is a list or tuple of four values [x_min, x_max, y_min, y_max]

    Returns:
    float: IOU value
    """
    x_min1, x_max1, y_min1, y_max1 = rect1
    x_min2, x_max2, y_min2, y_max2 = rect2

    # Calculate the (x, y) coordinates of the intersection rectangle
    x_min_inter = max(x_min1, x_min2)
    x_max_inter = min(x_max1, x_max2)
    y_min_inter = max(y_min1, y_min2)
    y_max_inter = min(y_max1, y_max2)

    # Calculate the area of the intersection rectangle
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    # Calculate the area of both rectangles
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Calculate the area of the union
    if union_type is None:
        union_area = area1 + area2 - inter_area
    elif union_type == 'area1':
        union_area = area1
    elif union_type == 'area2':
        union_area = area2

    # Compute the IOU
    iou = 1.0 * inter_area / union_area if union_area != 0 else 0.0

    return np.array(iou, dtype=np.float64)

def calculate_iom(rect1, rect2, union_type:Optional[str]=None):
    """
    Calculate the Intersection over Minimum (IOU) of two rectangles.
    """
    x_min1, x_max1, y_min1, y_max1 = rect1
    x_min2, x_max2, y_min2, y_max2 = rect2

    # Calculate the (x, y) coordinates of the intersection rectangle
    x_min_inter = max(x_min1, x_min2)
    x_max_inter = min(x_max1, x_max2)
    y_min_inter = max(y_min1, y_min2)
    y_max_inter = min(y_max1, y_max2)

    # Calculate the area of the intersection rectangle
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    # Calculate the area of both rectangles
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    union_area = min(area1, area2)

    # Compute the IOU
    iou = 1.0 * inter_area / union_area if union_area != 0 else 0.0

    return np.array(iou, dtype=np.float64)

def update_physics_material(scene):
    new_pm = PhysicsMaterial(
        prim_path = "/World/Physics_Materials/obj_physics_material",
        static_friction  = 0.5,
        dynamic_friction = 0.5,
        restitution  = 0
    )
    objs = scene.get_all_rigid_objects()
    for obj_name in objs:
        obj = scene.get_object_by_name(obj_name)
        obj.baselink.apply_visual_material(new_pm)


if __name__ == '__main__':
    args = parse_args()
    eval_res_save_path = os.path.join(args.result_dir, 'eval_res_' + timestamp_str() + '.json')
    object_result_dir = os.path.join(args.result_dir, 'object')
    sensor_output_dir = os.path.join(args.result_dir, 'pick_vis_tongverse')
    dataset_test_dir = args.dataset_test_dir

    ## set random seed
    seed = 2024
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = deepcopy(default_env_cfg)
    camcfg1 = deepcopy(default_camera_cfg)

    scene_cfg = cfg.scene
    agent_cfg = cfg.agent
    agent_cfg['agent'] = 'Mec_kinova'
    env = Env(cfg); env.reset()

    camcfg1.cam_params["name"] = "persp_cam"
    camcfg1.cam_params["parent"] = "/World"
    cam1 = Camera(camcfg1)
    cam2 = None; cam3 = None
    cur_scene_name = scene_cfg['name']

    from mk_sampler import MecKinovaSampler
    mksampler = MecKinovaSampler(device='cpu', num_fixed_points=1024, use_cache=True)
    
    eval_res = {}
    id_iter = tqdm(sorted(os.listdir(object_result_dir), key=lambda x:int(x.split('.')[0])), desc="{0: ^10}".format('ID'), leave=False, mininterval=0.1)
    for id_name in id_iter:
        sensor_output_id_path = os.path.join(sensor_output_dir, id)
        rmdir_if_exists(sensor_output_id_path)
        mkdir_if_not_exists(sensor_output_id_path, True)
        result_file_path = os.path.join(object_result_dir, id_name)
        test_file_path = os.path.join(dataset_test_dir, id + '.npy')

        ## result file
        with open(os.path.join(result_file_path), "r") as f:
            result_item = json.load(f) 
        
        ## the corresponding test set file
        test_data = np.load(test_file_path, allow_pickle=True).item()

        ## load data
        sce_name = test_data['scene']['name']
        obj_name = test_data['object']['name']
        T_eeo_se3 = SE3(matrix=test_data['task']['grasping_pose'])
        T_rot_z_se3 = SE3(matrix=np.array([
                [0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        )
        T_eeo_se3 = T_eeo_se3.__matmul__(T_rot_z_se3)
        place_area_bbox = test_data['task']['placement_area']['scene_placement_pc']['bbox_test']
        place_area_pc = np.array(test_data['task']['placement_area']['scene_placement_pc']['points_test'])
        traj_w = result_item['trajectory_w']
        config_smoothness = result_item['config_smoothness']
        if ((traj_w[-1][0] > 1e2) or (traj_w[-1][0] < -1e2)) and (config_smoothness == 0):
            continue

        ## update environment
        # agent_cfg['base_joint_pos_spawn_position'] = tuple(traj_w[-1][:3])
        scene_cfg['name'] = sce_name
        if cur_scene_name != sce_name:
            env.reload_scene(scene_cfg)
            cur_scene_name = sce_name
        # env.reload_agent(agent_cfg)
        agent = env.get_agent()

        ## update camera
        if cam2 is not None: 
            cam2.remove(); cam2 = None
        if cam3 is not None: 
            cam3.remove(); cam3 = None

        env.reset(); env.step()

        ## remove ceiling
        for ceiling in env.get_scene().get_objects_by_category("ceiling"):
            ceiling.set_visibility(False)
        for _ in range(100): env.sim.render()

        ## simulation
        cnt = 0
        while True:
            if not env.sim.is_playing():
                env.sim.render()
                continue
            if not LIGHT_ADJUSTMENT:
                cprint.fatal('Please Adjust the Light!')
                LIGHT_ADJUSTMENT = True
                scene = env.get_scene()
                update_physics_material(scene)
                env.reset()
                env.pause() 
            if cnt == START:
                scene = env.get_scene()
                agent = env.get_agent()
                obj = scene.get_object_by_name(obj_name)
                T_eew = mksampler.end_effector_pose(torch.tensor(traj_w[-1])).squeeze(0).clone().detach().cpu().numpy()
                T_eew_se3 = SE3(matrix=T_eew)
                T_ow_se3 = T_eew_se3.__matmul__(T_eeo_se3.inverse)
                obj.set_world_pose(T_ow_se3.xyz, T_ow_se3.so3.wxyz)
                # agent.set_joint_state(positions=torch.tensor([traj_w[-1] + [0.0, 0.0]]))
                for _ in range(10): env.step(); env.sim.render() 
                obj_init_min_x, obj_init_min_y, obj_init_min_z, obj_init_max_x, obj_init_max_y, obj_init_max_z = obj.get_aabb()
                # env.pause() 
            elif cnt == END:
                obj_xyz_1, obj_q_1 = obj.get_world_pose()
                T_ow_1_se3 = SE3(xyz=obj_xyz_1.clone().detach().cpu().numpy(), quaternion=obj_q_1.clone().detach().cpu().numpy())
                for _ in range(200): env.step()
                obj_xyz_2, obj_q_2 = obj.get_world_pose()
                T_ow_2_se3 = SE3(xyz=obj_xyz_2.clone().detach().cpu().numpy(), quaternion=obj_q_2.clone().detach().cpu().numpy())

                ## obj movement
                t_move = np.linalg.norm(T_ow_1_se3._xyz - T_ow_2_se3._xyz)
                r_move = np.abs(np.degrees((T_ow_1_se3.so3._quat * T_ow_2_se3.so3._quat.conjugate).radians))

                ## bbox IOU
                obj_final_min_x, obj_final_min_y, obj_final_min_z, obj_final_max_x, obj_final_max_y, obj_final_max_z = obj.get_aabb()
                place_area_x_min = np.array(-place_area_bbox["extents"][0] / 2 + place_area_bbox["transform"][0][-1], dtype=np.float64)
                place_area_x_max = np.array(place_area_bbox["extents"][0] / 2 + place_area_bbox["transform"][0][-1], dtype=np.float64)
                place_area_y_min = np.array(-place_area_bbox["extents"][1] / 2 + place_area_bbox["transform"][1][-1], dtype=np.float64)
                place_area_y_max = np.array(place_area_bbox["extents"][1] / 2 + place_area_bbox["transform"][1][-1], dtype=np.float64)

                ## compute iou
                iom_init = calculate_iom(
                    tuple([place_area_x_min, place_area_x_max, place_area_y_min, place_area_y_max]), 
                    tuple([obj_init_min_x, obj_init_max_x, obj_init_min_y, obj_init_max_y]),
                )
                iom_final = calculate_iom(
                    tuple([place_area_x_min, place_area_x_max, place_area_y_min, place_area_y_max]), 
                    tuple([obj_final_min_x, obj_final_max_x, obj_final_min_y, obj_final_max_y]),
                )
                ## save
                eval_res[id] = {
                    'iom_init': iom_init,
                    'iom_final': iom_final,
                    't_move(m)': t_move,
                    'r_move(degree)': r_move,
                    'obj_final_xyz': obj_xyz_2.clone().detach().cpu().numpy(),
                    'obj_final_quaternion': obj_q_2.clone().detach().cpu().numpy()
                }
                dict2json(eval_res_save_path, eval_res)
                for _ in range(100): env.sim.render() 
                break

            env.step()
            # cprint.info(f'count:{cnt}')
            cnt += 1

