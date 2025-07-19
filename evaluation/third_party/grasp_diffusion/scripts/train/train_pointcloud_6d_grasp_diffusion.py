import os
import copy
import configargparse
from se3dif.utils import get_root_src

import torch
from torch.utils.data import DataLoader

from se3dif import datasets, losses, summaries, trainer
from se3dif.models import loader

from se3dif.utils import load_experiment_specifications

from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules

base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))


def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--specs_file_dir', type=str, default=os.path.join(base_dir, 'params')
                   , help='root for saving logging')

    p.add_argument('--spec_file', type=str, default='multiobject_p_graspdif'
                   , help='root for saving logging')

    p.add_argument('--summary', type=bool, default=False
                   , help='activate or deactivate summary')

    p.add_argument('--saving_root', type=str, default=os.path.join(get_root_src(), 'logs')
                   , help='root for saving logging')

    p.add_argument('--models_root', type=str, default=root_dir
                   , help='root for saving logging')

    p.add_argument('--device',  type=str, default='cuda',)
    p.add_argument('--class_type', type=str, default='Mug')

    opt = p.parse_args()
    return opt


def main(opt):

    ## Load training args ##
    spec_file = os.path.join(opt.specs_file_dir, opt.spec_file)
    """
    /home/ysx/0_WorkSpace/4_Grasping_Pointcloud_Networks/1_Grasp_Diffusion/grasp_diffusion/scripts/train/
    params/multiobject_p_graspdif
    """

    args = load_experiment_specifications(spec_file) #! 模型参数
    """
    args:
    {'Description': ['This experiment trains jointly an SDF model and a SE(3) Grasp Energy'], 'exp_log_dir': 
    'multiobject_p_graspdif', 'single_object': False, 'TrainSpecs': {'batch_size': 2, 'num_epochs': 90000, 
    'steps_til_summary': 500, 'iters_til_checkpoint': 1000, 'epochs_til_checkpoint': 10}, 'NetworkArch': 
    'PointcloudGraspDiffusion', 'NetworkSpecs': {'feature_encoder': {'enc_dim': 132, 'in_dim': 3, 'out_dim': 7, 
    'dims': [512, 512, 512, 512, 512, 512, 512, 512], 'dropout': [0, 1, 2, 3, 4, 5, 6, 7], 'dropout_prob': 0.2, 
    'norm_layers': [0, 1, 2, 3, 4, 5, 6, 7], 'latent_in': [4], 'xyz_in_all': False, 'use_tanh': False, 
    'latent_dropout': False, 'weight_norm': True}, 'encoder': {'latent_size': 132, 'hidden_dim': 512}, 
    'points': {'n_points': 30, 'loc': [0.0, 0.0, 0.5], 'scale': [0.7, 0.5, 0.7]}, 'decoder': {'hidden_dim': 512}}, 
    'LearningRateSchedule': [{'Type': 'Step', 'Initial': 0.0005, 'Interval': 500, 'Factor': 0.5}, {'Type': 'Step', 
    'Initial': 0.001, 'Interval': 500, 'Factor': 0.5}, {'Type': 'Step', 'Initial': 0.001, 'Interval': 500, 
    'Factor': 0.5}], 'Losses': ['sdf_loss', 'projected_denoising_loss']}
    """

    # saving directories
    root_dir = opt.saving_root
    exp_dir  = os.path.join(root_dir, args['exp_log_dir'])
    args['saving_folder'] = exp_dir

    """
    opt: 训练参数
    Namespace(class_type='Mug', config_filepath=None, device='cuda', 
    models_root='/home/ysx/0_WorkSpace/4_Grasping_Pointcloud_Networks', 
    saving_root='/home/ysx/0_WorkSpace/4_Grasping_Pointcloud_Networks/1_Grasp_Diffusion/logs', 
    spec_file='multiobject_p_graspdif', 
    specs_file_dir='/home/ysx/0_WorkSpace/4_Grasping_Pointcloud_Networks/1_Grasp_Diffusion/
    grasp_diffusion/scripts/train/params', summary=False)
    """
    if opt.device =='cuda':
        if 'cuda_device' in args:
            cuda_device = args['cuda_device']
        else:
            cuda_device = 0
        device = torch.device('cuda:' + str(cuda_device) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    ## Dataset
    train_dataset = datasets.PointcloudAcronymAndSDFDataset(augmented_rotation=True, one_object=args['single_object'])
    train_dataloader = DataLoader(train_dataset, batch_size=args['TrainSpecs']['batch_size'], shuffle=True, drop_last=True)
    test_dataset = copy.deepcopy(train_dataset)
    test_dataset.set_test_data()
    test_dataloader = DataLoader(test_dataset, batch_size=args['TrainSpecs']['batch_size'], shuffle=True, drop_last=True)

    ## Model
    args['device'] = device
    model = loader.load_model(args)

    # Losses
    loss = losses.get_losses(args) # ['sdf_loss', 'projected_denoising_loss']
    loss_fn = val_loss_fn = loss.loss_fn

    ## Summaries
    summary = summaries.get_summary(args, opt.summary)

    ## Optimizer
    lr_schedules = get_learning_rate_schedules(args)
    optimizer = torch.optim.Adam([
            {
                "params": model.vision_encoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": model.feature_encoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": model.decoder.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ])

    # Train
    trainer.train(model=model.float(), train_dataloader=train_dataloader, epochs=args['TrainSpecs']['num_epochs'], model_dir= exp_dir,
                summary_fn=summary, device=device, lr=1e-4, optimizers=[optimizer],
                steps_til_summary=args['TrainSpecs']['steps_til_summary'],
                epochs_til_checkpoint=args['TrainSpecs']['epochs_til_checkpoint'],
                loss_fn=loss_fn, iters_til_checkpoint=args['TrainSpecs']['iters_til_checkpoint'],
                clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
                val_dataloader=test_dataloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)