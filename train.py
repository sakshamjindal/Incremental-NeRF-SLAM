from models.poses import LearnPose
import os

from pytorch_lightning import loggers
from opt import get_opts
import torch
from collections import defaultdict
from gradslam.datasets.tum import TUM

from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import *

# models
from models.nerf import *
from models.rendering import *
from models.poses import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

#lietorch
from lietorch import SO3, SE3, Sim3
print("import succesful")

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()

        self.hparams = hparams
        self.root_dir = self.hparams.root_dir
        self.sequences = "sequences.txt"
        self.img_wh = tuple(self.hparams.img_wh)
        self.train_rand_rows = 64
        self.train_rand_cols = 64
        self.near = 0.05
        self.far = 1
        self.W, self.H = tuple(self.hparams.img_wh)

        self.poses_to_train = [29]

        if self.hparams.lamda > 0:
            if self.hparams.depth_norm:
                self.loss = [
                    loss_dict['color'](coef=1),
                    loss_dict['depth_norm'](coef=1) 
                ]
            else:
                self.loss = [
                    loss_dict['color'](coef=1),
                    loss_dict['depth'](coef=1) 
                ]
        else:
            self.loss = [loss_dict['color'](coef=1)]

        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)

        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        self.nerf_coarse = NeRF()
        self.models = {'coarse': self.nerf_coarse}

        if self.hparams.freeze_nerf:
            self.models["coarse"].eval()
            for param in self.models["coarse"].parameters():
                param.requires_grad = False

        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        self.model_pose = LearnPose(num_cams = 1, learn_R = True, learn_t = True)

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models['fine'] = self.nerf_fine
            
            if self.hparams.freeze_nerf:
                self.models["fine"].eval()
                for param in self.models["fine"].parameters():
                    param.requires_grad = False
                    
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

        self._setup()

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def _setup(self):
        
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh),
                  'start': self.hparams.start,
                  'end' : self.hparams.end,
                  'period' : self.hparams.period,
                  'poses_to_train' : self.poses_to_train
                  }

        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

        self.all_poses = self.train_dataset.all_poses
        self.directions = self.train_dataset.directions
        num_poses_to_train = len(self.poses_to_train)
        self.model_pose = LearnPose(num_poses_to_train, learn_R=True, learn_t=True, init_c2w = self.all_poses)


    def configure_optimizers(self):
        if self.hparams.freeze_nerf:
            self.optimizer = get_optimizer(self.hparams, self.model_pose)
            scheduler = get_scheduler(self.hparams, self.optimizer)
        else:
            self.optimizer = get_optimizer(self.hparams, [self.models, self.model_pose])
            scheduler = get_scheduler(self.hparams, self.optimizer)
            
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):

        idx = batch['idx'].detach().cpu().numpy()[0]
        img = batch['rgbs']
        batch_size = img.shape[0]
        img = img[0]
        #print("posse:{}".format(SE3(self.all_poses).data))

        if self.hparams.lamda > 0:
            masks = batch['masks']
            depths = batch['depths']

        c2w = self.model_pose(idx)
        c2w = c2w[:3, :4]

        # sample pixel on an image and their rays for training
        r_id = torch.randperm(self.H, device = c2w.device)[:self.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(self.W, device = c2w.device)[:self.train_rand_cols]  # (N_select_cols)
        img = img[r_id][:, c_id]
        ray_selected_cam = self.directions[r_id][:, c_id].to(c2w.device)  # (N_select_rows, N_select_cols, 3)
        rays_o, rays_d = get_rays(ray_selected_cam, c2w) # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, self.near*torch.ones_like(rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], dim = 1)

        rgbs = img.view(-1, 3)

        results = self(rays)
        rgb_results = {k: v for k, v in results.items() if 'rgb' in k}
        loss_rgb = self.loss[0](rgb_results, rgbs)
        
        if self.hparams.lamda > 0:
            depth_results = {k: v for k, v in results.items() if 'depth' in k}
            loss_depth = self.loss[1](depth_results, depths, masks)
        
        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        
        if self.hparams.lamda > 0:
            loss = loss_rgb + self.hparams.lamda*loss_depth
        else:
            loss = loss_rgb

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/rgb_loss', loss_rgb)
        if self.hparams.lamda > 0:
            self.log('train/depth_loss', loss_depth)
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):

        idx = batch['idx'].detach().cpu().numpy()[0]
        img = batch['rgbs']
        img = img[0]

        if self.hparams.lamda > 0:
            masks = batch['masks']
            depths = batch['depths']

        c2w = self.model_pose(idx)
        c2w = c2w[:3, :4] 

        # sample pixel on an image and their rays for validations
        ray_selected_cam = self.directions.to(c2w.device) # (H, W, 3)
        rays_o, rays_d = get_rays(ray_selected_cam, c2w) # both (H*W, 3)
        rays = torch.cat([rays_o, rays_d, self.near*torch.ones_like(rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], dim = 1)
        
        if self.hparams.lamda > 0:
            depths = batch['depths'] #(h*w)
            masks = batch['masks']

        # rays = rays.squeeze() # (H*W, 3)
        rgbs = img.view(-1, 3) # (H*W, 3)
        results = self(rays)
        rgb_results = {k: v for k, v in results.items() if 'rgb' in k}
        if self.hparams.lamda > 0:
            depth_results  = {k: v for k, v in results.items() if 'depth' in k}

        log = {'val_rgb_loss': self.loss[0](rgb_results, rgbs)}

        if self.hparams.lamda > 0:
            log['val_depth_loss'] = self.loss[1](depth_results, depths, masks)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            if self.hparams.lamda > 0:
                depth_gt = visualize_depth(depths.view(H, W), resize = False) # (3, H, W)

            depth = visualize_depth(results[f'depth_{typ}'].view(H, W), resize = False) # resize to (640,320)
            stack = torch.stack([img_gt, img]) # (2, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_image_depth',
                                            stack, self.current_epoch)

            if self.hparams.lamda == 0.0:  
                self.logger.experiment.add_images('val/depth_images',
                                    depth.unsqueeze(0), self.current_epoch)

            
            if self.hparams.lamda > 0:
                stack = torch.stack([depth_gt, depth]) #(2, 3, H, W) 
                self.logger.experiment.add_images('val/depth_images',
                                    stack, self.current_epoch)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):

        mean_rgb_loss = torch.stack([x['val_rgb_loss'] for x in outputs]).mean()
        if self.hparams.lamda > 0:
            mean_depth_loss = torch.stack([x['val_depth_loss'] for x in outputs]).mean()
        
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/rgb_loss', mean_rgb_loss)
        if self.hparams.lamda > 0:
            self.log('val/depth_loss', mean_depth_loss)

        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(filepath=os.path.join(f'inv_ckpts/{hparams.exp_name}',
                                               '{epoch:d}'),
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=5)

    logger = TestTubeLogger(save_dir="inv_logs",
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch = hparams.val_frequency,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.gpus,
                      #gpus = [1],
                      accelerator='ddp' if len(hparams.gpus)>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if len(hparams.gpus)==1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
