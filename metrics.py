import torch
from kornia.losses import ssim as dssim
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import se3_log_map

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

def pose_metrics(c2w, gt_pose):

    pose_loss = torch.norm(se3_log_map(torch.unsqueeze((c2w@torch.inverse(gt_pose)).t(),0)))

    gt_pose = gt_pose[:3,:3].cpu()
    c2w = c2w[:3,:3].cpu()
    r1 = R.from_matrix(gt_pose)
    r2 = R.from_matrix(c2w)

    q1 = r1.as_quat()
    q1 = Quaternion(q1)
    q2 = r2.as_quat()
    q2 = Quaternion(q2)

    # geodesical distance given by logmap(quaternion)
    quat_dist = Quaternion.distance(q1, q2)

    quat_dist = torch.FloatTensor([quat_dist])

    return pose_loss, quat_dist
