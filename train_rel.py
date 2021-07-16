import os

from PIL.Image import init
import hyp

from pytorch_lightning import loggers
from opt import get_opts, dataset_path
import torch
from collections import defaultdict
from metrics import pose_metrics
from gradslam.datasets.tum import TUM

from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import *
from datasets.concat import ConcatDataset

# models
from models.nerf import *
from models.rendering import *
from models.poses import *
from models.poses import LearnPose

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
        self.train_rand_rows = 45
        self.train_rand_cols = 45
        self.near = 0.05
        self.far = 1
        self.W, self.H = tuple(self.hparams.img_wh)

        if self.hparams.mode == "f_to_g":
            self.f_to_g = True
            self.g_to_f = False
        elif self.hparams.mode == "g_to_f":
            self.g_to_f = True
            self.f_to_g = False
        elif self.hparams.mode == "both":
            self.f_to_g = True
            self.g_to_f = True
            
        if self.f_to_g:
            self.start_g = hyp.start_g
            self.period_g = hyp.period_g
            self.end_g = hyp.end_g
            self.nerf_g_weight_path = hyp.nerf_g_weight_path
            self.poses_to_train_f = self.hparams.poses_to_train_f
            self.poses_to_val_f = self.hparams.poses_to_val_f
            self.optimised_poses_f = self.hparams.optimised_poses_f
            self.nerf_f_pose_path = self.hparams.nerf_f_pose_path
            self.models_g = {}
    
        if self.g_to_f:
            self.start_f = hyp.start_f
            self.period_f = hyp.period_f
            self.end_f = hyp.end_f
            self.nerf_f_weight_path  = hyp.nerf_f_weight_path
            self.poses_to_train_g = self.hparams.poses_to_train_g
            self.poses_to_val_g = self.hparams.poses_to_val_g
            self.optimised_poses_g = self.hparams.optimised_poses_g
            self.nerf_g_pose_path = self.hparams.nerf_g_pose_path
            self.models_f = {}

        if self.hparams.lamda > 0:
            if self.hparams.depth_norm:
                self.loss = [loss_dict['color'](coef=1),loss_dict['depth_norm'](coef=1)]
            else:
                self.loss = [loss_dict['color'](coef=1),loss_dict['depth'](coef=1)]
        else:
            self.loss = [loss_dict['color'](coef=1)]

        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)

        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        nerf_types = ['coarse', 'fine']

        for nerf_type in nerf_types:

            if nerf_type == "fine" and not hparams.N_importance > 0:
                continue

            if nerf_type == "coarse" and not hparams.N_samples > 0:
                continue

            if self.g_to_f:
                if nerf_type == "coarse":
                    self.nerf_coarse_f = NeRF()
                    self.models_f['{}'.format(nerf_type)] =  self.nerf_coarse_f
                elif nerf_type == "fine":
                    self.nerf_fine_f = NeRF()
                    self.models_f['{}'.format(nerf_type)] =  self.nerf_fine_f
                
                if self.hparams.freeze_nerf:
                    self.models_f['{}'.format(nerf_type)].eval()
                    for param in self.models_f['{}'.format(nerf_type)].parameters():
                        param.requires_grad = False

                load_ckpt(self.models_f['{}'.format(nerf_type)], self.nerf_f_weight_path, 'nerf_{}'.format(nerf_type))

            if self.f_to_g:
                if nerf_type == "coarse":
                    self.nerf_coarse_g = NeRF()
                    self.models_g['{}'.format(nerf_type)] =  self.nerf_coarse_g
                elif nerf_type == "fine":
                    self.nerf_fine_g = NeRF()
                    self.models_g['{}'.format(nerf_type)] =  self.nerf_fine_g
                if self.hparams.freeze_nerf:
                    self.models_g['{}'.format(nerf_type)].eval()
                    for param in self.models_g['{}'.format(nerf_type)].parameters():
                        param.requires_grad = False

                load_ckpt(self.models_g['{}'.format(nerf_type)], self.nerf_g_weight_path, 'nerf_{}'.format(nerf_type))

        self._setup()

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, mode = "f_to_g"):

        if mode == "f_to_g":
            model = self.models_g
        elif mode == "g_to_f":
            model = self.models_f
        
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(model,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            False)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def _setup(self):
        
        dataset = dataset_dict[self.hparams.dataset_name]

        if self.f_to_g:
            kwargs_f = {
                'root_dir': dataset_path,
                'img_wh': tuple(self.hparams.img_wh),
                'start': self.start_g,
                'end' : self.end_g,
                'period' : self.period_g,
                'poses_to_train' : self.poses_to_train_f,
                'poses_to_val' : self.poses_to_val_f,
                'optimised_poses' : self.optimised_poses_f,
                'pose_params_path' : self.nerf_f_pose_path
            }

            self.train_dataset_f = dataset(split='train', **kwargs_f)
            self.val_dataset_f = dataset(split='val', **kwargs_f)
            self.directions = self.train_dataset_f.directions

        if self.g_to_f:
            kwargs_g = {
                'root_dir': dataset_path,
                'img_wh': tuple(self.hparams.img_wh),
                'start': self.start_f,
                'end' : self.end_f,
                'period' : self.period_f,
                'poses_to_train' : self.poses_to_train_g,
                'poses_to_val' : self.poses_to_val_g,
                'optimised_poses' : self.optimised_poses_g,
                'pose_params_path' : self.nerf_g_pose_path
            }

            self.train_dataset_g = dataset(split='train', **kwargs_g)

            initialise_rel_pose = True
            init_c2w = None
            if initialise_rel_pose:
                alt_pose = self.train_dataset_g[0]['alt_poses']
                gt_pose = self.train_dataset_g[2]['gt_poses']
                init_c2w = torch.unsqueeze(torch.mm(gt_pose, torch.inverse(alt_pose)), 0)

            self.val_dataset_g = dataset(split='val', **kwargs_g)
            self.directions = self.train_dataset_g.directions

        self.relative_pose = LearnPose(1, learn_R=True, learn_t=True, init_c2w = init_c2w)
        #load_ckpt(self.relative_pose, self.hparams.relative_pose_weight_path, 'relative_pose')

    def configure_optimizers(self):

        self.optimizer = get_optimizer(self.hparams, self.relative_pose)
        scheduler = get_scheduler(self.hparams, self.optimizer)  
            
        return [self.optimizer], [scheduler]

    def train_dataloader(self):

        loaders = {}
        
        if self.f_to_g:
            loaders['data_f'] = DataLoader(
                                    self.train_dataset_f,
                                    shuffle=True,
                                    num_workers=0,
                                    batch_size=1,
                                    pin_memory=True
                                )

            if not self.g_to_f:
                return loaders['data_f']

        if self.g_to_f:
            loaders['data_g'] = DataLoader(
                                    self.train_dataset_g,
                                    shuffle=True,
                                    num_workers=0,
                                    batch_size=1,
                                    pin_memory=True
                                )

            if not self.f_to_g:
                return loaders['data_g']

        if self.f_to_g and self.g_to_f:
            concat_dataset = ConcatDataset(
                self.train_dataset_f,
                self.train_dataset_g
            )
        
            return DataLoader(
                        concat_dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=1,
                        pin_memory=True
            )

    def val_dataloader(self):

        if self.f_to_g and not self.g_to_f:
            return DataLoader(self.val_dataset_f,
                            shuffle=False,
                            num_workers=0,
                            batch_size=1, # validate one image (H*W rays) at a time
                            pin_memory=True)
        else:
            return DataLoader(self.val_dataset_g,
                            shuffle=False,
                            num_workers=0,
                            batch_size=1, # validate one image (H*W rays) at a time
                            pin_memory=True) 

    def common_training_step(self, batch, mode = "f_to_g"):

        img = batch['rgbs'][0]
        alt_pose = batch['alt_poses'][0]
        gt_pose = batch['gt_poses'][0]

        if self.hparams.lamda > 0:
            masks = batch['masks'][0]
            depths = batch['depths'][0]
        
        if mode == "f_to_g":
            c2w = torch.mm(alt_pose, torch.inverse(self.relative_pose(0)))
        elif mode == "g_to_f":
            c2w = torch.mm(self.relative_pose(0), alt_pose)
        else:
            raise ValueError("Mode of relative pose transformation not defined properly")

        with torch.no_grad():
            pose_loss, _ = pose_metrics(c2w, gt_pose)

        c2w = c2w[:3, :4] 

        # sample pixel on an image and their rays for training
        r_id = torch.randperm(self.H, device = c2w.device)[:self.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(self.W, device = c2w.device)[:self.train_rand_cols]  # (N_select_cols)
        img = img[r_id][:, c_id]
        if self.hparams.lamda > 0:
            depths = depths[r_id][:, c_id]
            masks = masks[r_id][:, c_id]

        ray_selected_cam = self.directions[r_id][:, c_id].to(c2w.device)  # (N_select_rows, N_select_cols, 3)
        rays_o, rays_d = get_rays(ray_selected_cam, c2w) # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, self.near*torch.ones_like(rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], dim = 1)

        # reshaping for calculating losses
        rgbs = img.view(-1, 3)

        if self.hparams.lamda > 0:
            depths = depths.view(-1)
            masks = masks.view(-1)

        results = self(rays, mode)
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
            return pose_loss, loss, loss_rgb, loss_depth
        else:
            loss = loss_rgb
            return pose_loss,loss, loss_rgb, 0

    def training_step(self, batch, batch_nb):

        batch_x = {}
        if not ('data_f' in batch.keys() and 'data_g' in batch.keys()):
            if self.f_to_g and not self.g_to_f:
                batch_x['data_f'] = batch
            elif self.g_to_f and not self.f_to_g:
                batch_x['data_g'] = batch
        else:
            batch_x = batch      

        if self.f_to_g:
            pose_loss, loss, loss_rgb, loss_depth = self.common_training_step(batch_x['data_f'], mode = "f_to_g")

        if self.g_to_f:
            if self.f_to_g:
                pose_loss_f, loss_f, loss_rgb_f, loss_depth_f = self.common_training_step(batch_x['data_g'], mode = "g_to_f")
                pose_loss, loss, loss_rgb, loss_depth = pose_loss + pose_loss_f, loss + loss_f, loss_rgb + loss_rgb_f, loss_depth + loss_depth_f
            else:
                pose_loss, loss, loss_rgb, loss_depth = self.common_training_step(batch_x['data_g'], mode = "g_to_f")

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/rgb_loss', loss_rgb)
        if self.hparams.lamda > 0:
            self.log('train/depth_loss', loss_depth)
        self.log('train/loss', loss)
        #self.log('train/psnr', psnr_, prog_bar=True)
        self.log('train/poss_loss', pose_loss)

        return loss

    def validation_step(self, batch, batch_nb):

        if self.f_to_g:
            mode = "f_to_g"

        if self.g_to_f:
            mode = "g_to_f"

        img = batch['rgbs'][0]
        alt_pose = batch['alt_poses'][0]
        gt_pose = batch['gt_poses'][0]

        if self.hparams.lamda > 0:
            masks = batch['masks'][0]
            depths = batch['depths'][0]

        if mode == "f_to_g":
            c2w = torch.mm(alt_pose, torch.inverse(self.relative_pose(0, stage = "val")))
        elif mode == "g_to_f":
            c2w = torch.mm(self.relative_pose(0, stage="val"), alt_pose)

        with torch.no_grad():
            pose_loss, _ = pose_metrics(c2w, gt_pose)

        c2w = c2w[:3, :4]

        # sample pixel on an image and their rays for validations
        ray_selected_cam = self.directions.to(c2w.device) # (H, W, 3)
        rays_o, rays_d = get_rays(ray_selected_cam, c2w) # both (H*W, 3)
        rays = torch.cat([rays_o, rays_d, self.near*torch.ones_like(rays_o[:, :1]), self.far*torch.ones_like(rays_o[:, :1])], dim = 1)

        # rays = rays.squeeze() # (H*W, 3)
        rgbs = img.view(-1, 3) # (H*W, 3)
        rgbs = img.view(-1, 3)

        if self.hparams.lamda > 0:
            depths = depths.view(-1)
            masks = masks.view(-1)

        results = self(rays, mode)
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

        return log

    def validation_epoch_end(self, outputs):

        mean_rgb_loss = torch.stack([x['val_rgb_loss'] for x in outputs]).mean()
        if self.hparams.lamda > 0:
            mean_depth_loss = torch.stack([x['val_depth_loss'] for x in outputs]).mean()
        
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_pose_loss = torch.stack([x['pose_loss'] for x in outputs]).mean()

        self.log('val/rgb_loss', mean_rgb_loss)
        if self.hparams.lamda > 0:
            self.log('val/depth_loss', mean_depth_loss)

        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/pose_loss', mean_pose_loss, prog_bar=False)


def main(hparams):
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(filepath=os.path.join(f'rel_ckpts/{hparams.exp_name}',
                                               '{epoch:d}'),
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=5)

    logger = TestTubeLogger(save_dir="rel_logs",
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
