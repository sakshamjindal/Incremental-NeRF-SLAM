import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def visualize_depth(depth, cmap=cv2.COLORMAP_JET, return_PIL = False, resize = True):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    # mi = 0 # get minimum depth
    # ma = 1
    x = (x-mi)/max(ma-mi, 1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    # resizing for visual aid on tensorboard
    if resize:
        x_ = x_.resize((640,480), resample = Image.NEAREST)
    if return_PIL:
        return x_
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_
