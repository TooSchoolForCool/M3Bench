import argparse
import numpy as np
import torch
import trimesh
from example.transform import transform_pointcloud_torch
from scene import Scene
from se3dif.models.loader import load_model
from se3dif.samplers.grasp_samplers import Grasp_AnnealedLD
from se3dif.visualization import grasp_visualization
from visualize import visualize_point_cloud
from geometrout.transform import SE3, SO3
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':

    ## 1. Create scene and object
    scene = Scene('House_5')
    object = scene.get_link('Cup_55_link')
    # get object point cloud
    obj_pc, _ = trimesh.sample.sample_surface(object, 1000)
    # visualize_point_cloud(obj_pc, frame_size=0.1) #! 解开注释可以查看物体的点云

    ## 2. Sample grasping poses 
    device = 'cuda:0'
    # normalize object point clouds
    obj_pc *= 8.
    obj_pc_mean = np.mean(obj_pc, axis=0)
    obj_pc -= obj_pc_mean
    obj_pc = torch.as_tensor(obj_pc).float().to(device=device)
    # load model
    #! 如果需要修改 sample grapsing poses 的个数，请更改 num
    num = 100
    args = argparse.Namespace(device=device, model='grasp_dif_multi', batch=num)
    model_args = {'device': device, 'pretrained_model': args.model}
    grasp_energy_model = load_model(model_args)
    context = obj_pc[None, ...]
    grasp_energy_model.set_latent(context, batch=args.batch)
    # sample grasping poses
    generator = Grasp_AnnealedLD(grasp_energy_model, batch=args.batch, T=70, T_fit=70, k_steps=2, device=args.device)
    Hs = generator.sample()

    ## 3. Unnormalize object point clouds and Hs
    Hs = Hs.clone().detach().cpu().numpy()
    obj_pc = obj_pc.clone().detach().cpu().numpy()
    obj_pc = (obj_pc + obj_pc_mean) / 8.
    Hs[..., :3, -1] = (Hs[..., :3, -1] + obj_pc_mean) / 8.
    #! 解开注释可以查看生成的 grasping poses
    # grasp_visualization.visualize_grasps(Hs=Hs, p_cloud=obj_pc)

    ## 4. Process results
    scale = 0.05 #! 夹爪向外平移的距离通过 scale 控制，单位(m)
    Hs = torch.tensor(Hs)
    gripper_center_point = torch.tensor([[0, 0, 6.59999996e-02]])
    gripper_center_points = gripper_center_point.unsqueeze(0).repeat(num, 1, 1)
    gripper_center_points_transform = transform_pointcloud_torch(gripper_center_points, Hs)
    gripper_center_points_transform = gripper_center_points_transform.squeeze(1)
    Hs[..., :3, -1] += (Hs[..., :3, -1] - gripper_center_points_transform) / 6.59999996e-02 * scale
    Hs = Hs.clone().detach().cpu().numpy()
    #! 解开注释可以查看生成的 grasping poses
    grasp_visualization.visualize_grasps(Hs=Hs, p_cloud=obj_pc)

    ## 5. Convert to a usable form of VKC
    attach_trans = []
    attach_orient = []
    for i in range(num):
        H = Hs[i,...]
        xyz = H[:3, -1]
        q = R.from_matrix(H[:3, :3]).as_quat()
        q = np.array([q[3], q[0], q[1], q[2]]) # q_vkc = [w, x, y, z]
        attach_trans.append(xyz)
        attach_orient.append(q)
    #! VKC 需要的数据
    attach_trans = np.stack(attach_trans, axis=0) 
    attach_orient = np.stack(attach_orient, axis=0)