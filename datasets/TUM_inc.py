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
        initial_poses : torch.Tensor,
        sequences = "sequences.txt",
        split='train',
        img_wh=(640, 480),
        start: Optional[int] = None,
        period: Optional[int] = 1,
        end: Optional[int] = None,
        poses_to_train = [],
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

        self.poses = initial_poses

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
        all_gt_poses = all_gt_poses[self.start:self.end:self.period, :, :]
        # Step4 : Processing poses now to feed into the algorithm
        
        num_images = all_colors.shape[0]
        self.poses = self.poses.numpy() # (N_images, 3, 4) cam2world matrices
        self.poses = self.poses[:, :3].astype('float64') # (N_images, 3, 4) cam2world matrices
        all_gt_poses = all_gt_poses.numpy()
        all_gt_poses = all_gt_poses[:, :3].astype('float64') # (All_images, 3, 4) cam2world matrices


        ## COLMAP poses hasse rotation in form "right down front", change to "right up back"
        ## See https://github.com/bmild/nerf/issues/34
        #self.poses = np.concatenate([self.poses[..., 0:1], - self.poses[..., 1:3], self.poses[..., 3:4]], -1)
        all_gt_poses = np.concatenate([all_gt_poses[..., 0:1], - all_gt_poses[..., 1:3], all_gt_poses[..., 3:4]], -1)

        ## Center the poses
        # self.poses, avg_pose = center_poses(poses)
        # self.all_gt_poses, _ = center_poses(all_gt_poses, pose_avg = avg_pose)
        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center) # choose val image as the closest to
        #                                            # center image

        # Step 5: correct scale so that the max depth is little closer (less than) 1.0
        scale_factor = 3.59949474527
        self.scale_factor = scale_factor
        all_gt_poses[..., 3] /= scale_factor

        ## ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)

        ## convert from 3x4 form to 4x4 form
        all_gt_poses = convert3x4_4x4(all_gt_poses) # (All_images, 4, 4)
        self.poses = convert3x4_4x4(self.poses) # (All_images, 4, 4)

        self.all_rgbs = []
        self.all_poses = []
        self.all_gt_poses = []
        self.all_depths = []
        self.all_masks = []

        if self.split == "train":
            poses_to_consider = self.poses_to_train
        elif self.split == "val":
            poses_to_consider = self.poses_to_val

        for i in range(len(poses_to_consider)):
#                 print("pose_conisder",poses_to_consider)
                pose_index = poses_to_consider[i]

                ## Initialisation of poses to optimise
                c2w = torch.FloatTensor(self.poses[pose_index]).view(1, 4, 4) # (4, 4)
                ## Ground truth poses
                gt_pose = torch.FloatTensor(all_gt_poses[pose_index]).view(1, 4, 4) # (4, 4)
                ## images and depths
                img = all_colors[pose_index]
                depth = all_depths[pose_index]            

                img = Image.fromarray(img.numpy().astype('uint8')).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.permute(1, 2, 0) # (h, w, 3) RGB
                self.all_rgbs.append(img)
                self.all_poses.append(c2w)
                self.all_gt_poses.append(gt_pose)

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
#         print("all_poses",self.all_poses.shape)
        self.all_gt_poses = torch.cat(self.all_gt_poses, dim = 0)
        assert len(self.all_rgbs) > 0 and len(self.all_poses) > 0 and len(self.all_depths) > 0 and len(self.all_masks) > 0

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
#             print("pose_all",self.all_poses)
        elif self.split == "val":
            pose_number = self.poses_to_val[idx]
            positional_index = self.poses_to_train.index(pose_number)

        sample = {
            'idx' : positional_index,
            'poses': self.all_poses[idx],
            'gt_poses': self.all_gt_poses[idx],
            'rgbs': self.all_rgbs[idx],
            'depths' : self.all_depths[idx],
            'masks': self.all_masks[idx]
        }

        return sample
