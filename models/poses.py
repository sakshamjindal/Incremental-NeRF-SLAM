import torch
import torch.nn as nn
from .lie_group_helpers import *
from lietorch import SO3, SE3, Sim3


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)
#         self.v = nn.Parameter(torch.tensor([[0., 0., 1., 0., 0.,0.]]),requires_grad=learn_R)
#         print(self.v.device)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
#         print("posse:{}".format(SE3.exp(self.v).matrix()))
#         c2w = SE3.exp(self.v).matrix()
        
        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]
#             c2w = c2w.mul(self.init_c2w[cam_id]).matrix()

        return c2w