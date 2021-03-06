from models.poses import LearnPose
import os

from pytorch_lightning import loggers
from opt import get_opts, dataset_path
import torch
from collections import defaultdict
from metrics import pose_metrics
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
        kwargs = {'root_dir': dataset_path,
                  'img_wh': tuple(self.hparams.img_wh),
                  'start': self.hparams.start,
                  'end' : self.hparams.end,
                  'period' : self.hparams.period,
                  'poses_to_train' : self.hparams.poses_to_train,
                  'poses_to_val' : self.hparams.poses_to_val,
                  'initial_poses' : self.hparams.initial_poses,
                  }

        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

        self.all_poses = self.train_dataset.all_initial_poses
        self.directions = self.train_dataset.directions
        num_poses_to_train = len(self.hparams.poses_to_train)
        
        if self.hparams.pose_optimization:
            self.model_pose = LearnPose(num_poses_to_train, learn_R=True, learn_t=True, init_c2w = self.all_poses)
        else:
            self.model_pose = LearnPose(num_poses_to_train, learn_R=False, learn_t=False, init_c2w = self.all_poses)
            self.model_pose.eval()


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
        img = batch['rgbs'][0]
        gt_pose = batch['gt_poses'][0]

        if self.hparams.lamda > 0:
            masks = batch['masks'][0]
            depths = batch['depths'][0]

        c2w = self.model_pose(idx)

        # Calculating pose loss metric - pose loss, quartenian distances, geodesical distances
        with torch.no_grad():
            pose_loss, quat_distance = pose_metrics(c2w, gt_pose)

        c2w = c2w[:3, :4] 

        # sample pixel on an image and their rays for training
        r_id = torch.randperm(self.H, device = c2w.device)[:self.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(self.W, device = c2w.device)[:self.train_rand_cols]  # (N_select_cols)
        img = img[r_id][:, c_id]
        depths = depths[r_id][:, c_id]
        masks = masks[r_id][:, c_id]
        ray_selected_cam = self.directions[r_id][:, c_id].to(c2w.device)  # (N_select_rows, N_select_cols, 3)
        rays_o, rays_d = get_rays(ray_selected_cam, c2w) # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, self.near*torch.ones_like(rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], dim = 1)

        # reshaping for calculating losses
        rgbs = img.view(-1, 3)
        depths = depths.view(-1)
        masks = masks.view(-1)

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
        self.log('train/pose_loss',pose_loss)
        self.log('train/quat_distance', quat_distance)
        if self.hparams.lamda > 0:
            self.log('train/depth_loss', loss_depth)
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):

        idx = batch['idx'].detach().cpu().numpy()[0]
        img = batch['rgbs'][0]
        gt_pose = batch['gt_poses'][0]

        if self.hparams.lamda > 0:
            masks = batch['masks'][0]
            depths = batch['depths'][0]

        c2w = self.model_pose(idx, stage="val")

        with torch.no_grad():
            pose_loss, quat_distance = pose_metrics(c2w, gt_pose)

        c2w = c2w[:3, :4] 

        # sample pixel on an image and their rays for validations
        ray_selected_cam = self.directions.to(c2w.device) # (H, W, 3)
        rays_o, rays_d = get_rays(ray_selected_cam, c2w) # both (H*W, 3)
        rays = torch.cat([rays_o, rays_d, self.near*torch.ones_like(rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], dim = 1)

        # rays = rays.squeeze() # (H*W, 3)
        rgbs = img.view(-1, 3) # (H*W, 3)
        rgbs = img.view(-1, 3)
        depths = depths.view(-1)
        masks = masks.view(-1)

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
        log['pose_loss'] = pose_loss
        log['quat_distance'] = quat_distance

        return log

    def validation_epoch_end(self, outputs):

        mean_rgb_loss = torch.stack([x['val_rgb_loss'] for x in outputs]).mean()
        if self.hparams.lamda > 0:
            mean_depth_loss = torch.stack([x['val_depth_loss'] for x in outputs]).mean()
        
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_pose_loss = torch.stack([x['pose_loss'] for x in outputs]).mean()
        mean_quat_distance = torch.stack([x['quat_distance'] for x in outputs]).mean()

        self.log('val/rgb_loss', mean_rgb_loss)
        if self.hparams.lamda > 0:
            self.log('val/depth_loss', mean_depth_loss)

        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/pose_loss', mean_pose_loss, prog_bar=False)
        self.log('val/quat_distance', mean_quat_distance, prog_bar=False)


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
    breakpoint()

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
