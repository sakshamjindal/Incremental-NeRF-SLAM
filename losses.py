from torch import nn
import torch

class ColorLoss(nn.Module):

    def __init__(
        self, 
        coef=1,
        loss_type="L2"
    ):
        super().__init__()
        self.coef = coef

        if loss_type == "L2":
            self.loss = nn.MSELoss(reduction='mean')
        elif loss_type == "L1":
            self.loss = nn.L1Loss(reduction="mean")

    def forward(self, inputs, targets):
        
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef*(loss)

class MaskedMSELoss(nn.Module):
    def __init__(self, reduction = "mean"):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, inputs, targets, masks):

        if masks is None:
            out = self.loss(inputs, targets)
        else:
            out = torch.sum(((inputs-targets)*masks)**2.0) / torch.sum(masks)
            
        return out


class CustomLoss(nn.Module):
    def __init__(
        self, 
        loss_type="L2"
    ):
        super().__init__()
        self.loss_type = loss_type


    def forward(
        self,
        inputs,
        targets, 
        masks,
        depth_variance
    ):

        # if not depth_variance:
        #     depth_variance = torch.ones_like(inputs)

        # if not masks:
        #     masks = torch.ones_like(inputs)

        if self.loss_type == "L1":
            e = (torch.abs(inputs - targets)*masks)
            e = e/torch.sqrt(depth_variance)
            e = torch.sum(e)/masks.sum()
        elif self.loss_type == "L2":
            e = torch.sum((((inputs-targets)*masks)**2.0)/depth_variance) / torch.sum(masks)


        return e           

class DepthLoss(nn.Module):

    def __init__(
        self, 
        coef=1,
        loss_type="L2"
    ):
        super().__init__()
        self.coef = coef

        self.loss = MaskedMSELoss()

    def forward(self, inputs, targets, masks):
        
        loss = self.loss(inputs['depth_coarse'], targets, masks)
        if 'depth_fine' in inputs:
            loss += self.loss(inputs['depth_fine'], targets, masks)

        return self.coef*(loss)

class CustomDepthLoss(nn.Module):

    def __init__(
        self, 
        coef=1,
        loss_type="L2"
    ):
        super().__init__()
        self.coef = coef

        self.loss = CustomLoss("L2")

    def forward(self, inputs, targets, masks):
        
        loss = self.loss(inputs['depth_coarse'], targets, masks, inputs["depth_variance_coarse"])
        if 'depth_fine' in inputs:
            loss += self.loss(inputs['depth_fine'], targets, masks, inputs["depth_variance_fine"])

        return self.coef*(loss)
        
loss_dict = {'color': ColorLoss,
             'depth': DepthLoss,
             'depth_norm' : CustomDepthLoss}



# from torch import nn
# import torch

# class ColorLoss(nn.Module):

#     def __init__(
#         self, 
#         coef=1,
#         loss_type="L2"
#     ):
#         super().__init__()
#         self.coef = coef

#         if loss_type == "L2":
#             self.loss = nn.MSELoss(reduction='mean')
#         elif loss_type == "L1":
#             self.loss = nn.L1Loss(reduction="mean")

#     def forward(self, inputs, targets):
        
#         loss = self.loss(inputs['rgb_coarse'], targets)
#         if 'rgb_fine' in inputs:
#             loss += self.loss(inputs['rgb_fine'], targets)

#         return self.coef*(loss)

# class MaskedMSELoss(nn.Module):
#     def __init__(self, reduction = "mean"):
#         super(MaskedMSELoss, self).__init__()
#         self.loss = nn.MSELoss(reduction=reduction)

#     def forward(self, inputs, targets, masks):

#         if masks is None:
#             out = self.loss(inputs, targets)
#         else:
#             out = torch.sum(((inputs-targets)*masks)**2.0) / torch.sum(masks)
            
#         return out


# class CustomLoss(nn.Module):
#     def __init__(
#         self, 
#         loss_type="L2"
#     ):
#         super().__init__()
#         self.loss_type = loss_type


#     def forward(
#         self,
#         inputs,
#         targets, 
#         masks,
#         depth_variance
#     ):

#         if not depth_variance:
#             depth_variance = torch.ones_like(inputs)

#         if not masks:
#             masks = torch.ones_like(inputs)

#         if self.loss_type == "L1":
#             e = (torch.abs(inputs - targets)*masks)
#             e = e/torch.sqrt(depth_variance)
#             e = torch.sum(e)/masks.sum()
#         elif self.loss_type == "L2":
#             e = ((inputs - targets)*masks)
#             e = e/torch.sqrt(depth_variance)
#             e = torch.sum(e)/masks.sum()

#         return e           

# class DepthLoss(nn.Module):

#     def __init__(
#         self, 
#         coef=1,
#         loss_type="L2"
#     ):
#         super().__init__()
#         self.coef = coef

#         self.loss = CustomLoss(loss_type)

#     def forward(self, inputs, targets, masks, depth_variance):
        
#         loss = self.loss(inputs['depth_coarse'], targets)
#         if 'depth_fine' in inputs:
#             loss += self.loss(inputs['depth_fine'], targets, depth_variance)

#         return self.coef*(loss)
        
# loss_dict = {'color': ColorLoss,
#              'depth': DepthLoss}