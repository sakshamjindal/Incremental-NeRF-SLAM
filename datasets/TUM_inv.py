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


def center_poses(poses, pose_avg = None):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    if pose_avg is None:
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
        sequences = "sequences.txt",
        split='train',
        img_wh=(640, 480),
        start: Optional[int] = None,
        period: Optional[int] = 1,
        end: Optional[int] = None,
        poses_to_train = [],
        initial_poses = [],
        poses_to_val = []
    ):
        """
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        start : starting number of the frame
        end : end number of the frame
        period : periodicity of selection of frame
        """
        self.root_dir = root_dir
        self.sequences = sequences
        self.split = split
        self.img_wh = img_wh
        self.start = start
        self.end = end
        self.period = period

        self.poses_to_train = sorted(poses_to_train)
        self.poses_to_val = sorted(poses_to_val)
        self.initial_poses = eval(initial_poses)

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
        dataset = TUM(self.root_dir, sequences = self.sequences, seqlen = 100, start = 56)
        all_colors, all_depths, all_intrinsics, all_gt_poses, _, __, ___ = dataset[0]

        # Step 3: subset poses based on start, end and period
        # Right now doing using input parameters but the pipeline will be tweaked if
        # we are doing keyframe selection
        poses = all_gt_poses[self.start:self.end:self.period, :, :]
        # Step4 : Processing poses now to feed into the algorithm
        #num_images = all_colors.shape[0]
        poses = poses.numpy() # (N_images, 3, 4) cam2world matrices
        poses = poses[:, :3].astype('float64') # (N_images, 3, 4) cam2world matrices
        all_gt_poses = all_gt_poses.numpy()
        all_gt_poses = all_gt_poses[:, :3].astype('float64') # (All_images, 3, 4) cam2world matrices


        ## COLMAP poses hasse rotation in form "right down front", change to "right up back"
        ## See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        all_gt_poses = np.concatenate([all_gt_poses[..., 0:1], - all_gt_poses[..., 1:3], all_gt_poses[..., 3:4]], -1)

        ## Center the poses
        self.poses, avg_pose = center_poses(poses)
        self.all_gt_poses, _ = center_poses(all_gt_poses, pose_avg = avg_pose)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image

        # Step 5: correct scale so that the max depth is little closer (less than) 1.0
        scale_factor = 3.59949474527
        self.scale_factor = scale_factor
        self.all_gt_poses[..., 3] /= scale_factor

        ## ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)

        ## convert from 3x4 form to 4x4 form
        self.all_gt_poses = convert3x4_4x4(self.all_gt_poses) # (All_images, 4, 4)

        self.all_rgbs = []
        self.all_initial_poses = []
        self.gt_poses = []
        self.all_depths = []
        self.all_masks = []

        if self.split == "train":
            poses_to_consider = self.poses_to_train
        elif self.split == "val":
            poses_to_consider = self.poses_to_val

        for i in range(len(poses_to_consider)):

                pose_index = poses_to_consider[i]

                ## Initialisation of poses to optimise
                c2w = torch.FloatTensor(self.all_gt_poses[self.initial_poses[pose_index]]).view(1, 4, 4) # (4, 4)
                ## Ground truth poses
                gt_pose = torch.FloatTensor(self.all_gt_poses[pose_index]).view(1, 4, 4) # (4, 4)
                ## images and depths
                img = all_colors[pose_index]
                depth = all_depths[pose_index]            

                img = Image.fromarray(img.numpy().astype('uint8')).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.permute(1, 2, 0) # (h, w, 3) RGB
                self.all_rgbs.append(img)
                self.all_initial_poses.append(c2w)
                self.gt_poses.append(gt_pose)

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

        self.all_initial_poses = torch.cat(self.all_initial_poses, dim = 0)
        self.gt_poses = torch.cat(self.gt_poses, dim = 0)
        assert len(self.all_rgbs) > 0 and len(self.all_initial_poses) > 0 and len(self.all_depths) > 0 and len(self.all_masks) > 0

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rgbs)
        if self.split == 'val':
            return 1

    def __getitem__(self, idx):

        if self.split == "train":
            pose_number = self.poses_to_train[idx]
            positional_index = self.poses_to_train.index(pose_number)
        elif self.split == "val":
            pose_number = self.poses_to_val[idx]
            positional_index = self.poses_to_train.index(pose_number)

        sample = {
            'idx' : positional_index,
            'poses': self.all_initial_poses[idx],
            'gt_poses': self.gt_poses[idx],
            'rgbs': self.all_rgbs[idx],
            'depths' : self.all_depths[idx],
            'masks': self.all_masks[idx]
        }

        return sample
