import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import math
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


class TUMDataset(Dataset):
    def __init__(self, 
        root_dir,
        sequences = "sequences.txt",
        split='train',
        img_wh=(640, 480),
        spheric_poses=False,
        start: Optional[int] = None,
        period: Optional[int] = 1,
        end: Optional[int] = None,
        val_num=1
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
        self.val_num = max(1, val_num) # at least 1
        self.start = start
        self.end = end
        self.period = period
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

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        # near_original = np.percentile(depths[depths>0],0.1)
        # scale_factor = near_original*0.75 # 0.75 is the default parameter
        #                                   # the nearest depth is at 1/0.75=1.33
        scale_factor = 3.59949474527
        #scale_factor = 1
        self.scale_factor = scale_factor
        # depths /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            self.all_depths = []
            self.all_masks = []
            for i in range(num_images):
                if i == val_idx: # exclude the val image
                    continue
                c2w = torch.FloatTensor(self.poses[i])
                img = self.colors[i]
                depth = self.depths[i]            

                img = Image.fromarray(img.numpy().astype('uint8')).convert('RGB')
                # assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                #     f'''{image_path} has different aspect ratio than img_wh, 
                #         please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]

                depth = depth.squeeze(-1).numpy()
                import cv2
                depth = cv2.resize(depth, (self.img_wh), interpolation = cv2.INTER_NEAREST)
                mask = 1 - ((depth).astype('uint8'))==0        
                mask = torch.Tensor(mask)
                mask = mask.view(1, -1).permute(1,0)
                depth = depth/scale_factor
                depth = torch.Tensor(depth)
                depth = depth.view(1, -1).permute(1,0)
                self.all_depths += depth
                self.all_masks += mask

                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                if not self.spheric_poses:
                    near, far = 0.05, 1
                    # rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                    #                               self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max()) # focus on central object only

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
            self.all_depths = torch.cat(self.all_depths, 0) # ((N_image-1), 1)
            self.all_masks = torch.cat(self.all_masks, 0)
        
        elif self.split == 'val':
            print('val image index is', val_idx)
            names = names.replace(' ','').split(',')
            print('val img name is ', names[val_idx])
            self.val_idx = val_idx

        else: # for testing, create a parametric rendering path
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                pass

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        if self.split == 'test_train':
            return len(self.poses)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'depths' : self.all_depths[idx],
                      'masks': self.all_masks[idx],
                      'poses' : self.poses}

        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[self.val_idx])
            elif self.split == 'test_train':
                c2w = torch.FloatTensor(self.poses[idx])
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0.05, 1
                # rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                #                               self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split in ['val', 'test_train']:
                if self.split == 'val':
                    idx = self.val_idx
                
                img = self.colors[idx]          
                img = Image.fromarray(img.numpy().astype('uint8')).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img

                depth = self.depths[idx]
                depth = depth.squeeze(-1).numpy()
                depth = cv2.resize(depth, (self.img_wh), interpolation = cv2.INTER_NEAREST)
                mask = 1 - ((depth).astype('uint8'))==0        
                mask = torch.Tensor(mask)
                mask = mask.view(-1)
                depth = depth/self.scale_factor
                depth = torch.Tensor(depth)
                depth = depth.view(-1)
                sample['depths'] = depth
                sample['masks'] = mask


        return sample
        