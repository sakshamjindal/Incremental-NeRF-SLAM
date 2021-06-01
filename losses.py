import torch
from torch import nn

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        # print("[INFO] RGB LOSS CALC", loss)
        return self.coef * loss

class DepthLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef

    def forward(self, inputs, targets_depth):
        # mask = (targets_depth>0)
        loss = torch.mean(torch.abs(inputs['depth_coarse'] - targets_depth) )# / (inputs['depth_std_coarse']+1))
        if 'rgb_fine' in inputs:
            loss += torch.mean(torch.abs(inputs['depth_fine'] - targets_depth) )# / (inputs['depth_fine']+1))
        # loss = torch.mean(torch.abs(inputs['depth_coarse'] - targets_depth) / (inputs['depth_std_coarse']+1e-6))
        # if 'rgb_fine' in inputs:
        #     loss += torch.mean(torch.abs(inputs['depth_fine'] - targets_depth) / (inputs['depth_fine']+1e-6))
        # print("[INFO] DEPTH LOSS CALC", loss)
        return self.coef * loss

class JointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss1 = ColorLoss(coef=1)
        self.loss2 = DepthLoss(coef=0.2)

    def forward(self, inputs, target_rgb, target_depths):
        # print("[INFO SHAPES]", inputs['depth_coarse'].shape, inputs['rgb_coarse'].shape, inputs['depth_std_coarse'].shape, target_depths.shape)
        # print("[INFO RANGE]", torch.min(inputs['depth_coarse']), torch.max(inputs['depth_coarse']), torch.min(target_depths), torch.max(target_depths))

        loss = self.loss1(inputs, target_rgb) + self.loss2(inputs, target_depths)
        return loss
    
loss_dict = {'color': ColorLoss, 'depth': DepthLoss, 'joint': JointLoss}
