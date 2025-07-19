import os
import torch
import torch.nn as nn
import numpy as np

from se3dif import models


from se3dif.utils import get_pretrained_models_src, load_experiment_specifications
pretrained_models_dir = get_pretrained_models_src()


def load_model(args):
    """
    {'Description': ['This experiment trains jointly an SDF model and a SE(3) Grasp Energy'], 
    'exp_log_dir': 'multiobject_p_graspdif', 'single_object': False, 
    'TrainSpecs': {'batch_size': 2, 'num_epochs': 90000, 'steps_til_summary': 500, 
    'iters_til_checkpoint': 1000, 'epochs_til_checkpoint': 10}, 'NetworkArch': 'PointcloudGraspDiffusion', 
    'NetworkSpecs': {'feature_encoder': {'enc_dim': 132, 'in_dim': 3, 'out_dim': 7, 
    'dims': [512, 512, 512, 512, 512, 512, 512, 512], 'dropout': [0, 1, 2, 3, 4, 5, 6, 7], 
    'dropout_prob': 0.2, 'norm_layers': [0, 1, 2, 3, 4, 5, 6, 7], 'latent_in': [4], 
    'xyz_in_all': False, 'use_tanh': False, 'latent_dropout': False, 'weight_norm': True}, 
    'encoder': {'latent_size': 132, 'hidden_dim': 512}, 'points': {'n_points': 30, 'loc': [0.0, 0.0, 0.5], 
    'scale': [0.7, 0.5, 0.7]}, 'decoder': {'hidden_dim': 512}}, 
    'LearningRateSchedule': [{'Type': 'Step', 'Initial': 0.0005, 'Interval': 500, 'Factor': 0.5}, 
    {'Type': 'Step', 'Initial': 0.001, 'Interval': 500, 'Factor': 0.5}, 
    {'Type': 'Step', 'Initial': 0.001, 'Interval': 500, 'Factor': 0.5}], 
    'Losses': ['sdf_loss', 'projected_denoising_loss'], 
    'saving_folder': '/home/ysx/0_WorkSpace/4_Grasping_Pointcloud_Networks/1_Grasp_Diffusion/logs/multiobject_p_graspdif', 
    'device': device(type='cuda', index=0)}
    """
    if 'pretrained_model' in args:
        model_args = load_experiment_specifications(os.path.join(pretrained_models_dir,
                                                                      args['pretrained_model']))
        args["NetworkArch"] = model_args["NetworkArch"]
        args["NetworkSpecs"] = model_args["NetworkSpecs"]

    if args['NetworkArch'] == 'GraspDiffusion':
        model = load_grasp_diffusion(args)
    elif args['NetworkArch'] == 'PointcloudGraspDiffusion': #! √
        model = load_pointcloud_grasp_diffusion(args)


    if 'pretrained_model' in args:
        model_path = os.path.join(pretrained_models_dir, args['pretrained_model'], 'model.pth')

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        if args['device'] != 'cpu':
            model = model.to(args['device'], dtype=torch.float32)

    elif 'saving_folder' in args:
        load_model_dir = os.path.join(args['saving_folder'], 'checkpoints', 'model_current.pth')
        try:
            if args['device'] == 'cpu':
                model.load_state_dict(torch.load(load_model_dir, map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(load_model_dir))
        except:
            pass

    return model


def load_grasp_diffusion(args):
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    lat_params = params['latent_codes']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.LatentCodes(num_scenes=lat_params['num_scenes'], latent_size=lat_params['latent_size'])
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= lat_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    # 3D Points
    if 'loc' in points_params:
        points = models.points.get_3d_pts(n_points = points_params['n_points'],
                            loc=np.array(points_params['loc']),
                            scale=np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )

    model = models.GraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder=energy_net, points=points).to(device)
    return model


def load_pointcloud_grasp_diffusion(args):
    """
    args:
    {'Description': ['This experiment trains jointly an SDF model and a SE(3) Grasp Energy'], 
    'exp_log_dir': 'multiobject_p_graspdif', 'single_object': False, 
    'TrainSpecs': {'batch_size': 2, 'num_epochs': 90000, 'steps_til_summary': 500, 
    'iters_til_checkpoint': 1000, 'epochs_til_checkpoint': 10}, 'NetworkArch': 'PointcloudGraspDiffusion', 
    'NetworkSpecs': {'feature_encoder': {'enc_dim': 132, 'in_dim': 3, 'out_dim': 7, 
    'dims': [512, 512, 512, 512, 512, 512, 512, 512], 'dropout': [0, 1, 2, 3, 4, 5, 6, 7], 
    'dropout_prob': 0.2, 'norm_layers': [0, 1, 2, 3, 4, 5, 6, 7], 'latent_in': [4], 
    'xyz_in_all': False, 'use_tanh': False, 'latent_dropout': False, 'weight_norm': True}, 
    'encoder': {'latent_size': 132, 'hidden_dim': 512}, 'points': {'n_points': 30, 'loc': [0.0, 0.0, 0.5], 
    'scale': [0.7, 0.5, 0.7]}, 'decoder': {'hidden_dim': 512}}, 
    'LearningRateSchedule': [{'Type': 'Step', 'Initial': 0.0005, 'Interval': 500, 'Factor': 0.5}, 
    {'Type': 'Step', 'Initial': 0.001, 'Interval': 500, 'Factor': 0.5}, 
    {'Type': 'Step', 'Initial': 0.001, 'Interval': 500, 'Factor': 0.5}], 
    'Losses': ['sdf_loss', 'projected_denoising_loss'], 
    'saving_folder': '/home/ysx/0_WorkSpace/4_Grasping_Pointcloud_Networks/1_Grasp_Diffusion/logs/multiobject_p_graspdif', 
    'device': device(type='cuda', index=0)}
    """
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    #! 点云输入的 encoder，对应论文中的 shape codes，输出 z，out_features = 132
    vision_encoder = models.vision_encoder.VNNPointnet2(out_features=v_enc_params['latent_size'], device=device)
    # Geometry encoder
    #! SE(3)的 encoder，输出 x_0
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'], # 132
            latent_size=v_enc_params['latent_size'], # 132
            dims=feat_enc_params['dims'], # [512, 512, 512, 512, 512, 512, 512, 512]
            out_dim=feat_enc_params['out_dim'], # 7
            dropout=feat_enc_params['dropout'], # [0, 1, 2, 3, 4, 5, 6, 7]
            dropout_prob=feat_enc_params['dropout_prob'], # 0.2
            norm_layers=feat_enc_params['norm_layers'], # [0, 1, 2, 3, 4, 5, 6, 7]
            latent_in=feat_enc_params["latent_in"], # [4]
            xyz_in_all=feat_enc_params["xyz_in_all"], # False
            use_tanh=feat_enc_params["use_tanh"], # False
            latent_dropout=feat_enc_params["latent_dropout"], # False
            weight_norm=feat_enc_params["weight_norm"] # True
        )
    # 3D Points
    #! 可能需要根据 robotiq gripper 的参数修改 'n_points' 'loc' 'scale'
    if 'loc' in points_params: # [0.0, 0.0, 0.5]
        points = models.points.get_3d_pts(n_points = points_params['n_points'], # 30
                            loc=np.array(points_params['loc']), # [0.0, 0.0, 0.5]
                            scale=np.array(points_params['scale'])) # [0.7, 0.5, 0.7]
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    # Energy Based Model
    in_dim = points_params['n_points']*feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )

    model = models.GraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder=energy_net, points=points).to(device)
    return model