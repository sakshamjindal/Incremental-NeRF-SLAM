from pytorch_lightning import callbacks
from models.poses import LearnPose
import os

from pytorch_lightning import loggers
from opt import get_opts, dataset_path, get_hparams
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

# import hyperparameters
import hyp

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger


class NeRFSystem(LightningModule):
    def __init__(
        self,
        hparams,
        inital_poses,
        start = 0,
        period = 0,
        end = 1,
        poses_to_train = [],
        poses_to_val = [],
        freeze_nerf = True,
        pose_optimization = True,
    ):
        
        super(NeRFSystem, self).__init__()

        self.hparams = hparams
        self.start = start
        self.period = period
        self.end = end
        self.poses_to_train = poses_to_train
        self.poses_to_val = poses_to_val
        self.inital_poses = inital_poses
        self.pose_optimization = pose_optimization
        self.freeze_nerf = freeze_nerf

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

        if freeze_nerf:
            self.models["coarse"].eval()
            for param in self.models["coarse"].parameters():
                param.requires_grad = False

        #load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models['fine'] = self.nerf_fine
            
            if freeze_nerf:
                self.models["fine"].eval()
                for param in self.models["fine"].parameters():
                    param.requires_grad = False
                    
            #load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

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
                  'start': self.start,
                  'end' : self.end,
                  'period' : self.period,
                  'poses_to_train' : self.poses_to_train,
                  'poses_to_val' : self.poses_to_val,
                  'initial_poses' : self.inital_poses,
                  }

        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

        self.all_poses = self.train_dataset.all_poses
        self.directions = self.train_dataset.directions
        num_poses_to_train = len(self.poses_to_train)
        
        if self.pose_optimization:
            self.model_pose = LearnPose(num_poses_to_train, learn_R=True, learn_t=True, init_c2w = self.all_poses)
        else:
            self.model_pose = LearnPose(num_poses_to_train, learn_R=False, learn_t=False, init_c2w = self.all_poses)
            self.model_pose.eval()


    def configure_optimizers(self):
        if self.freeze_nerf:
            self.optimizer = get_optimizer(self.hparams, self.model_pose)
            scheduler = get_scheduler(self.hparams, self.optimizer)
        else:
            self.optimizer = get_optimizer(self.hparams, [self.models, self.model_pose])
            scheduler = get_scheduler(self.hparams, self.optimizer)
            
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
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

def get_trainer(exp_name, hparams, max_epochs, resume_from_checkpoint = None):

    checkpoint_callback = ModelCheckpoint(
                                filepath=os.path.join(f'inc_ckpts/{exp_name}','{epoch:d}'), 
                                monitor='val/psnr',
                                mode='max',
                                save_top_k=5
                        )
    early_stop_callback = EarlyStopping(
                            monitor='val/psnr',
                            min_delta=0.5,
                            patience=2,
                            verbose=False,
                            mode='max'
                        )
    
    if resume_from_checkpoint is None:
        logger = TestTubeLogger(
                    save_dir="inc_logs",
                    name=exp_name,
                    debug=False,
                    create_git_tag=False,
                    log_graph=False
                )
        num_sanity_val_step = 1
        callbacks = [checkpoint_callback, early_stop_callback]
    else:
        logger = None
        num_sanity_val_step = 0
        callbacks = None

    trainer = Trainer(
                max_epochs=max_epochs, 
                resume_from_checkpoint = resume_from_checkpoint,
                check_val_every_n_epoch = hparams.val_frequency,
                callbacks = callbacks, 
                logger=logger,
                weights_summary=None,
                progress_bar_refresh_rate=1,
                gpus=list(hparams.gpus),
                accelerator='ddp' if len(hparams.gpus)>1 else None,
                num_sanity_val_steps=num_sanity_val_step,
                benchmark=True,
                profiler="simple" if len(hparams.gpus)==1 else None
            )

    return trainer



def main(hparams):

    import hyp
    if hyp.DEBUG:
        from opt import get_hparams
        hparams = get_hparams("commands.txt")

    if not hyp.saved_params:
        # (Initialise training from scractch)
        trainer = get_trainer(exp_name = "nerf_0", hparams = hparams, max_epochs = 2000, resume_from_checkpoint = "/scratch/saksham/nerf_pl/inc_ckpts/nerf_0/epoch=1999.ckpt")
        # (initial)
        poses = torch.FloatTensor([[ 1.,  0.,  0.,  0.], [ 0., -1.,  0.,  0.], [ 0.,  0., -1.,  0.], [ 0.,  0.,  0.,  1.]]).unsqueeze(0)
        model_nerf = NeRFSystem(start = 0, period = 1, end = 1, poses_to_train = [0], poses_to_val = [0], 
                        inital_poses = poses, freeze_nerf = False, pose_optimization = False, hparams=hparams)
        trainer.fit(model_nerf)
        checkpoint_nerf_ = model_nerf.state_dict()
        initial_index = 1
    else:
        # (Resume training from a saved dump)
        params = torch.load(hyp.saved_params)
        initial_index = params["index"] + 1
        checkpoint_nerf_ = params["nerf_weights"]
        poses = params["poses"]
        print("Resuming incremental training from index : {}".format(initial_index))

    for i in range(initial_index,20):

        # pose optimisation first
        trainer = get_trainer(exp_name = "pose_opt_{}".format(i), hparams = hparams, max_epochs = 500)
        poses = torch.cat([poses, poses[i-1].unsqueeze(0)], dim = 0)
        poses_to_train = [i]
        poses_to_val = [i]
        model_pose = NeRFSystem(start = 0, period = 1, end = i + 1, poses_to_train = poses_to_train, poses_to_val = poses_to_val, 
                       inital_poses = poses, freeze_nerf = True, pose_optimization = True, hparams=hparams)
        load_ckpt(model_pose, checkpoint_nerf_, 'nerf_coarse')
        load_ckpt(model_pose, checkpoint_nerf_, 'nerf_fine')
        trainer.fit(model_pose)
        checkpoint_pose_ = model_pose.state_dict()

        # fine-tune the nerf after pose optimization
        trainer = get_trainer(exp_name = "nerf_{}".format(i), hparams = hparams, max_epochs = 500)
        assert len(model_pose.model_pose.state_dict()["r"]) == 1

        pose_model = LearnPose(1, learn_R=False, learn_t=False, init_c2w = poses[i-1].unsqueeze(0))
        pose_model.eval()
        load_ckpt(pose_model, checkpoint_pose_ , 'model_pose')
        
        with torch.no_grad():
            optimized_pose = pose_model(0)

        poses[i] = optimized_pose

        if i >= 5 :
            poses_to_train = list(np.linspace(1, i, 5, dtype=int))
        else:
            poses_to_train = [index for index in range(1, i+1)]
        
        print("Finetuning Nerf for poses : {} ".format(poses_to_train))

        poses_to_val = [i]
        model_nerf = NeRFSystem(
                        start = 0, period = 1, end = i + 1, poses_to_train = poses_to_train, poses_to_val = poses_to_val, 
                        inital_poses = poses, freeze_nerf = False, pose_optimization = True, hparams=hparams
                    )
        load_ckpt(model_nerf, checkpoint_nerf_, 'nerf_coarse')
        load_ckpt(model_nerf, checkpoint_nerf_, 'nerf_fine')
        trainer.fit(model_nerf)
        checkpoint_nerf_ = model_nerf.state_dict()

        pose_model = LearnPose(len(poses_to_train), learn_R=False, learn_t=False, init_c2w = poses[poses_to_train])
        pose_model.eval()
        load_ckpt(pose_model, checkpoint_nerf_ , 'model_pose')

        with torch.no_grad():
            for ind, pose in enumerate(poses_to_train):
                poses[pose] = pose_model(ind)

        save = {
            "index" : i, 
            "nerf_weights" : checkpoint_nerf_,
            "poses" : poses
        }

        if not os.path.isdir("inc_nerf_dumps/{}".format(hyp.Exp_name)):
            os.mkdir("inc_nerf_dumps/{}".format(hyp.Exp_name))

        torch.save(save, "inc_nerf_dumps/{}/dumps_{}".format(hyp.Exp_name, i))



if __name__ == '__main__':

    if hyp.DEBUG:
        from opt import get_hparams
        hparams = get_hparams("commands.txt")
    else:
        hparams = get_opts()
        
    main(hparams)




# pseudo code
'''
Steps
1) initially take one frame and train a nerf with identity matrix as the first pose
2) add one frame and optimise for pose using the above nerf weigths
3) optimise for pose + nerf weights 
4) repeat above steps in loop

Pseduo Code

- poses = []
- trainer = Trainer()
- init_dataset = dataset(start = 0, end = 0, poses_to_train = [0], poses_to_val=[0)
  model = Nerf(init_dataset, optimise_pose = False, freeze_nerf = False)
  Trainer.fit(model)
  poses.append(identity_poses)

- for i in range(1:):
    
    # pose optimisation first
    poses.append(poses[i-1])
    dataset = dataset(start = 0, end = i, poses_to_train = [i], poses_to_val[i], poses = poses)
    model_pose = Nerf(optimise_pose = True, freeze_nerf = True, dataset)
    model_pose.load_state_dict(model.state_dict)

    #fine tune nerf
    dataset = dataset(start = 0, end = 1, poses_to_train = [:i], poses_to_val[:i])
    model_nerf = Nerf(optmise_pose = True, freeze_nerf= False)
    model_nerf.load_state_dict(model_pose.state_dict)

'''