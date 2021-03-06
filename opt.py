import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # Common across pipeline

    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        #choices=['blender', 'llff', 'llff_tum', 'tum'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')
    parser.add_argument('--start', type=int, default=0, 
                        help='number of the starting frame')
    parser.add_argument('--end', type=int, default=None, 
                        help='number of the ending frame')
    parser.add_argument('--period', type=int, default=1, 
                        help='periodicity to select the frame') 
    parser.add_argument('--val_frequency', type=int, default=1, 
                        help='periodicity to select the frame') 
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--lamda', type=float, default=0.0,
                        help='factor for depth loss')
    parser.add_argument('--depth_norm', default=False, action="store_true",
                        help='use for normalizing depth loss')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')  
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--gpus', nargs="+", type = int, default=[0],
                        help='number of gpus')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    ## params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ## params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ## params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')


    # Used for inverse nerf computations
    parser.add_argument('--poses_to_train', nargs="+", type = int, default=[29],
                        help='nposes to train in inverse nerf')
    parser.add_argument('--poses_to_val', nargs="+", type = int, default=[29],
                        help='poses to validate in inverse nerf')
    parser.add_argument('--initial_poses', type = str, default="",
                        help='dictionary of poses to initialise with')
    parser.add_argument('--freeze_nerf', default=False, action="store_true",
                        help='freeze nerf while optimizing for poses')
    parser.add_argument('--pose_optimization', default=False, action="store_true",
                        help='enable pose optimization')      
    parser.add_argument('--pose_params_path', type=str, default=None,
                        help='pretrained checkpoint to load the poses(including optimizers, etc)')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained model weight to load (do not load optimizers, etc)')
    
    
    # Used for relative nerf pose computations
    parser.add_argument('--mode', type=str, default='f_to_g',
                        help='relative nerf mode')
                        
    parser.add_argument('--poses_to_train_f', nargs="+", type = int, default=[29],
                        help='nposes to train in inverse nerf')
    parser.add_argument('--poses_to_val_f', nargs="+", type = int, default=[29],
                        help='poses to validate in inverse nerf')
    parser.add_argument('--optimised_poses_f', nargs="+", type = int, default=[29],
                        help='number of gpus')
    parser.add_argument('--nerf_f_pose_path', type=str, default=None,
                        help='pretrained checkpoint to load the poses(including optimizers, etc)')

    parser.add_argument('--poses_to_train_g', nargs="+", type = int, default=[29],
                        help='nposes to train in inverse nerf')
    parser.add_argument('--poses_to_val_g', nargs="+", type = int, default=[29],
                        help='poses to validate in inverse nerf')
    parser.add_argument('--optimised_poses_g', nargs="+", type = int, default=[29],
                        help='number of gpus')
    parser.add_argument('--nerf_g_pose_path', type=str, default=None,
                        help='pretrained checkpoint to load the poses(including optimizers, etc)')
    # parser.add_argument('--relative_pose_weight_path', type=str, default=None,
    #                     help='pretrained checkpoint to load the relative poses(including optimizers, etc)')

    return parser.parse_args()

def get_hparams(path):
    from attrdict import AttrDict
    import json
    with open(path, 'r') as f: 
        hparams = json.load(f)
    hparams = AttrDict(hparams)

    return hparams


import socket
server = (socket.gethostname())

if "Sakshams-MacBook" in server:
    dataset_path = "/Users/sakshamjindal/Coding-Workspace/datasets/tum/"
elif "blue" in server:
    dataset_path = "/scratch/saksham/datasets/tum/"
elif "neon" in server:
    dataset_path = "/scratch/saksham/datasets/tum/"
elif "gnode" in server:
    dataset_path = "/home2/saksham.jindal/datasets/tum"
else:
    raise ValueError("server name incorrect")