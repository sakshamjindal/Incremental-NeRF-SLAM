import torch
from torch.utils.data import Dataset
import glob
import math
import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms as T
from typing import Optional

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from models.poses import LearnPose
from utils import load_ckpt

from gradslam.datasets.tum import TUM

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def scale_poses(poses):
    scale = 8.669891269074654
    trans = np.array([ 0.20023559, -0.11089467,  0.01612732])
    rot = np.array([[ 0.99997401, -0.00337696,  0.00636965],[ 0.00319304,  0.99958392,  0.02866674],[-0.00646381, -0.02864566,  0.99956873]])

    for i in range(len(poses)):
        poses[i, :-1,-1] = scale * rot.dot(poses[i, :-1,-1]) + trans
        poses[i, :3, :3]= rot.dot(poses[i, :3, :3])

    return poses

def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


class TUMDataset(Dataset):
    def __init__(self, 
        root_dir,
        pose_params_path,
        sequences = "sequences.txt",
        split='train',
        img_wh=(640, 480),
        spheric_poses=False,
        start: Optional[int] = None,
        period: Optional[int] = 1,
        end: Optional[int] = None,
        poses_to_train = [],
        poses_to_val = [],
        optimised_poses = []
    ):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        start : starting number of the frame
        end : end number of the frame
        period : periodicity of selection of frame
        """
        self.root_dir = root_dir
        self.sequences = sequences
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.start = start
        self.end = end
        self.period = period
        self.poses_to_train = sorted(poses_to_train)
        self.poses_to_val = sorted(poses_to_val)
        self.optimised_poses = optimised_poses
        
        self.poses_to_train = [x - self.start for x in self.poses_to_train]
        self.poses_to_val = [x - self.start for x in self.poses_to_val]
        self.optimised_poses = [x - self.start for x in self.optimised_poses]

        self.num_poses = len(self.optimised_poses)
        pose_optimisation = False
        if not pose_optimisation:
            self.model_pose = LearnPose(self.num_poses, learn_R=False, learn_t=False, init_c2w = torch.zeros((self.num_poses, 4, 4)))
            self.model_pose = self.model_pose.cpu()
            self.model_pose.eval()

            for param in self.model_pose.parameters():
                param.requires_grad = False

        load_ckpt(self.model_pose, pose_params_path, 'model_pose')
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # Step 1: rescale focal length according to training resolution
        focal = 525
        H = 480
        W = 640
        self.focal = focal * self.img_wh[0]/W

        # Step 2: read poses from TUM dataloader
        # read extrinsics (of successfully reconstructed images)
        seqlen = math.ceil(((self.end - self.start)/self.period))
        dataset = TUM(self.root_dir, sequences = self.sequences, seqlen = seqlen, start = 56 + self.start, dilation = self.period - 1)
        self.colors, self.depths, intrinsics, poses, transforms, names, timestamps = dataset[0]
        num_images = self.colors.shape[0]
        poses = poses.numpy() # (N_images, 3, 4) cam2world matrices
        
        scale_pose = False
        if scale_pose:
            poses = scale_poses(poses)

        poses = poses[:, :3].astype('float64') # (N_images, 3, 4) cam2world matrices

        # Step 3 read bounds
        # COLMAP poses has rotation in form "right down fraont", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        
        #Center the pose
        self.poses, _ = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image

        # Step 3: correct scale so that the max depth is little closer (less than) 1.0
        scale_factor = 3.59949474527
        #scale_factor = 1
        self.scale_factor = scale_factor
        # depths /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)

        # convert from 3x4 form to 4x4 form
        self.poses = convert3x4_4x4(self.poses)  # (N, 4, 4)

            
        # if self.split == 'train': # create buffer of all rays and rgb data
        #                           # use first N_images-1 to train, the LAST is val
        self.all_rgbs = []
        self.all_poses = []
        self.all_depths = []
        self.all_masks = []

        for i in range(num_images):

            if i in self.optimised_poses:

                c2w = torch.FloatTensor(self.poses[i]).view(1, 4, 4) # (4, 4)
                img = self.colors[i]
                depth = self.depths[i]            

                img = Image.fromarray(img.numpy().astype('uint8')).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.permute(1, 2, 0) # (h, w, 3) RGB
                self.all_rgbs.append(img)
                self.all_poses.append(c2w)

                depth = depth.squeeze(-1).numpy()
                depth = cv2.resize(depth, (self.img_wh), interpolation = cv2.INTER_NEAREST)
                mask = 1 - ((depth).astype('uint8'))==0        
                mask = torch.Tensor(mask)
                mask = mask.view(self.img_wh[1], self.img_wh[0], 1) # (h, w, 1)
                depth = depth/scale_factor
                depth = torch.Tensor(depth)
                depth = depth.view(self.img_wh[1], self.img_wh[0], 1) # h, w, 1)

                self.all_depths.append(depth)
                self.all_masks.append(mask)


        self.all_poses = torch.cat(self.all_poses, dim = 0)
        assert len(self.all_rgbs) > 0 and len(self.all_poses) > 0 and len(self.all_depths) > 0 and len(self.all_masks) > 0

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.poses_to_train)
        if self.split == 'val':
            return len(self.poses_to_val)

    def __getitem__(self, idx):
        
        if self.split == 'train': # use data in the buffers
            train_idx = self.poses_to_train[idx]
            positional_index = self.optimised_poses.index(train_idx)
            sample = {
                'idx' : idx,
                'poses': self.all_poses[positional_index],
                'alt_poses' : self.model_pose(positional_index),
                'rgbs': self.all_rgbs[positional_index],
                'depths' : self.all_depths[positional_index],
                'masks': self.all_masks[positional_index]
            }

        else:
            if self.split == 'val':
                val_idx = self.poses_to_val[idx]
                positional_index = self.optimised_poses.index(val_idx)

                return {
                    'idx' : positional_index,
                    'poses': self.all_poses[positional_index],
                    'alt_poses' : self.model_pose(positional_index),
                    'rgbs': self.all_rgbs[positional_index],
                    'depths' : self.all_depths[positional_index],
                    'masks': self.all_masks[positional_index]
                }
            else:
                raise ValueError("split of the dataset should be either train or val")

        return sample
        
