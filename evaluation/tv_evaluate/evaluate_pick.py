from datetime import datetime
import json
import argparse
import shutil
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

#! NOTE: 如果要保存图片，请在命令行中添加 --save_image 参数，并执行以下操作
#! 进入isaac sim时迅速进入点击暂停，调整灯光，demo_light和stage_light

"""
python evaluate_pick.py --result_dir /your_ws_path/m3bench/results/pick/timestamp --dataset_test_dir /your_data_path/pick/test 
"""

class NumpyArrayEncoder(json.JSONEncoder):
    """
    Python dictionaries can store ndarray array types, but when serialized by dict into JSON files, 
    ndarray types cannot be serialized. In order to read and write numpy arrays, 
    we need to rewrite JSONEncoder's default method. 
    The basic principle is to convert ndarray to list, and then read from list to ndarray.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

SCENE_UPDATE = 50
CLOSE_GRIPPER = 150
LIFT = 250
STOP = 350
LIGHT_ADJUSTMENT = False

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
        T_ow = SE3(matrix=test_data['transformation_matrix']['T_ow'])
        traj_w = result_item['trajectory_w']

        ## update environment
        agent_cfg['base_joint_pos_spawn_position'] = tuple(traj_w[-1][:3])
        scene_cfg['name'] = sce_name
        if cur_scene_name != sce_name:
            env.reload_scene(scene_cfg)
            cur_scene_name = sce_name
        env.reload_agent(agent_cfg)
        agent = env.get_agent()

        ## update camera
        if cam2 is not None: 
            cam2.remove(); cam2 = None
        if cam3 is not None: 
            cam3.remove(); cam3 = None
        # camera 2
        camcfg2 = deepcopy(default_camera_cfg)
        camcfg2.output_dir = os.path.join(sensor_output_id_path)
        camcfg2.resolution = (2000, 2000)
        agent_baselink = f"{agent.prim_path}/base_link"
        camcfg2.cam_params["name"] = "pick_right_view"
        camcfg2.cam_params["parent"] = agent_baselink
        camcfg2.cam_params["focal_length"] = 7
        camcfg2.annotators["semantic_segmentation"] = False
        camcfg2.annotators["CameraParams"] = False
        cam2 = Camera(camcfg2)
        cam2.set_local_pose(translation=[-0.31, -2.0, 0.5], orientation=[0.70711, 0, 0, -0.70711])
        # camera 3
        camcfg3 = deepcopy(default_camera_cfg)
        camcfg3.output_dir = os.path.join(sensor_output_id_path)
        camcfg3.resolution = (2000, 2000)
        agent_baselink = f"{agent.prim_path}/base_link"
        camcfg3.cam_params["name"] = "pick_left_view"
        camcfg3.cam_params["parent"] = agent_baselink
        camcfg3.cam_params["focal_length"] = 7
        camcfg3.annotators["semantic_segmentation"] = False
        camcfg3.annotators["CameraParams"] = False
        cam3 = Camera(camcfg3)
        cam3.set_local_pose(translation=[-0.31, 2.0, 0.5], orientation=[0.70711, 0, 0, 0.70711])
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
                env.pause() 
            if cnt == SCENE_UPDATE:
                scene = env.get_scene()
                agent = env.get_agent()
                obj = scene.get_object_by_name(obj_name)
                obj.set_world_pose(T_ow.xyz, T_ow.so3.wxyz)
                agent.set_joint_state(positions=torch.tensor([traj_w[-1] + [0.0, 0.0]]))
                env.step()
                # env.pause() 
            elif cnt == CLOSE_GRIPPER:
                if args.save_image:
                    env.step(); cam2.save(); cam3.save()
                agent.gripper_controller.close()
                env.step()
                # for _ in range(100): env.sim.render() 
                if args.save_image:
                    env.step(); cam2.save(); cam3.save()
            elif cnt == LIFT:
                ## init grasping pose
                init_cfg = agent.get_joint_state().positions
                T_eew_init = mksampler.end_effector_pose(init_cfg[:-2]).squeeze(0).clone().detach().cpu().numpy()
                T_eew_init_se3 = SE3(matrix=T_eew_init)
                obj_xyz_init, obj_q_init = obj.get_world_pose()
                T_ow_init_se3 = SE3(xyz=obj_xyz_init.clone().detach().cpu().numpy(), quaternion=obj_q_init.clone().detach().cpu().numpy())
                H_init_se3 = T_ow_init_se3.inverse.__matmul__(T_eew_init_se3)

                for t in np.arange(1, 0.7, -0.05):
                    action = ArticulationAction(joint_positions=[traj_w[-1][4] * t], joint_indices=[4])
                    agent.apply_action(action)
                    if args.save_image:
                        env.step(); cam2.save(); cam3.save()
                
                ## final grasping pose
                final_cfg = agent.get_joint_state().positions
                T_eew_final = mksampler.end_effector_pose(final_cfg[:-2]).squeeze(0).clone().detach().cpu().numpy()
                T_eew_final_se3 = SE3(matrix=T_eew_final)
                obj_xyz_final, obj_q_final = obj.get_world_pose()
                T_ow_final_se3 = SE3(xyz=obj_xyz_final.clone().detach().cpu().numpy(), quaternion=obj_q_final.clone().detach().cpu().numpy())
                H_final_se3 = T_ow_final_se3.inverse.__matmul__(T_eew_final_se3)

                #! t_dist <= 3cm 视为成功
                t_dist = np.linalg.norm(H_init_se3._xyz - H_final_se3._xyz)
                r_dist = np.abs(np.degrees((H_init_se3.so3._quat * H_final_se3.so3._quat.conjugate).radians))

                cprint(f'{id}: t_dist is {t_dist}(m), r_dist is {r_dist}(°)')
                # env.pause() 
            elif cnt == STOP:
                eval_res[id] = {
                    'H_init': H_init_se3.matrix,
                    'H_final': H_final_se3.matrix,
                    't_dist(m)': t_dist,
                    'r_dist(degree)': r_dist
                }
                dict2json(eval_res_save_path, eval_res)
                if args.save_image:
                    env.step(); cam2.save(); cam3.save()
                for _ in range(100): env.sim.render() 
                break

            env.step()
            # cprint.info(f'count:{cnt}')
            cnt += 1