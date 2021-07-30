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
from datasets.concat import ConcatDataset

# models
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

# Import model - siren or original
if hyp.model_type == "siren":
    from models.nerf import *
else:
    from models.nerf_origin import *

class NeRFSystem(LightningModule):
    def __init__(
        self,
        hparams,
        inital_poses,
        poses_to_train = [],
        poses_to_val = [],
        keyframe_indexes = [],
        freeze_nerf = True,
        pose_optimization = True,
        relative_pose_optimization = True
    ):
        
        super(NeRFSystem, self).__init__()

        self.hparams = hparams
        self.poses_to_train = poses_to_train
        self.poses_to_val = poses_to_val
        self.inital_poses = inital_poses
        self.keyframe_indexes = keyframe_indexes
        self.freeze_nerf = freeze_nerf
        self.pose_optimization = pose_optimization
        self.relative_pose_optimization = relative_pose_optimization

        self.img_wh = tuple(self.hparams.img_wh)
        self.train_rand_rows = 50
        self.train_rand_cols = 50
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

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models['fine'] = self.nerf_fine
            
            if freeze_nerf:
                self.models["fine"].eval()
                for param in self.models["fine"].parameters():
                    param.requires_grad = False

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
                            False)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def _setup(self):
        
        dataset = dataset_dict[self.hparams.dataset_name]

        # do relative pose optimisation only when all items of pose_to_train_g are present in common_frames_f_g
        # relative_pose_optimization will be set to False, after nerf starts training beyond common frames
        self.relative_pose_optimization = all(item in hyp.commom_frame_f_g for item in self.poses_to_train) and self.relative_pose_optimization

        if self.relative_pose_optimization:

            trained_poses = torch.load(hyp.saved_f)["poses"] # load the saved poses of previous nerf
            pose_indexes = torch.load(hyp.saved_f)["poses_trained"] # global indexes of saved poses
            poses_to_train_f = self.poses_to_train # we only want to load poses which we are optimising for in the present nerf

            kwargs_f = {'root_dir': dataset_path,
                        'img_wh': tuple(self.hparams.img_wh),
                        'poses_to_train' : poses_to_train_f ,
                        'poses_to_val' : poses_to_train_f ,
                        'initial_poses' : trained_poses,
                        'pose_indexes' : pose_indexes # global indexes of poses in initial poses
                    }

            self.train_dataset_f = dataset(split='train', **kwargs_f)
            self.val_dataset_f = dataset(split='val', **kwargs_f)

        kwargs_g = {'root_dir': dataset_path,
                  'img_wh': tuple(self.hparams.img_wh),
                  'poses_to_train' : self.poses_to_train,
                  'poses_to_val' : self.poses_to_val,
                  'initial_poses' : self.inital_poses,
                  'pose_indexes' : self.keyframe_indexes # global indexes of poses in initial poses
                  }

        self.train_dataset_g = dataset(split='train', **kwargs_g)
        self.val_dataset_g = dataset(split='val', **kwargs_g)

        self.all_poses = self.train_dataset_g.all_poses
        self.directions = self.train_dataset_g.directions
        num_poses_to_train = len(self.poses_to_train)

        if self.relative_pose_optimization:
            # get positional index of start in the previous nerf.
            # for example: if start =  15 and previous nerf was trained on keyframes : [5, 10, 15, 18]
            # then positional index of the starting point in previous nerf is 3
            # use this positional index to obtain the pose of starting point in previous nerf
            positional_index = pose_indexes.index(hyp.start_g)
            assert positional_index==15 # test
            relative_init_c2w = trained_poses[positional_index].view(1,4,4) # check
        
            self.relative_pose = LearnPose(1, learn_R=True, learn_t=True, init_c2w = relative_init_c2w)

        if self.pose_optimization:
            self.model_pose = LearnPose(num_poses_to_train, learn_R=True, learn_t=True, init_c2w = self.all_poses)
        else:
            self.model_pose = LearnPose(num_poses_to_train, learn_R=False, learn_t=False, init_c2w = self.all_poses)
            self.model_pose.eval()

    def configure_optimizers(self):

        optimizer_models = []
        if not self.freeze_nerf:
            optimizer_models.append(self.models)

        if self.pose_optimization:
            optimizer_models.append(self.model_pose)

        if self.relative_pose_optimization:
            optimizer_models.append(self.relative_pose)

        self.optimizer = get_optimizer(self.hparams, optimizer_models)
        scheduler = get_scheduler(self.hparams, self.optimizer)

        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        loaders = {}

        loaders['data_g'] = DataLoader(
                                self.train_dataset_g,
                                shuffle=True,
                                num_workers=0,
                                batch_size=1,
                                pin_memory=False
                            )
        
        if not self.relative_pose_optimization:
            return loaders['data_g']

        loaders['data_f'] = DataLoader(
                                self.train_dataset_f,
                                shuffle=True,
                                num_workers=0,
                                batch_size=1,
                                pin_memory=False
                            )

        concat_dataset = ConcatDataset(
            self.train_dataset_f,
            self.train_dataset_g
        )
        
        return DataLoader(
                    concat_dataset,
                    shuffle=True,
                    num_workers=0,
                    batch_size=1,
                    pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset_g,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=False
                )
    

    def common_training_step(self, batch, mode):
        idx = batch['idx'].detach().cpu().numpy()[0]
        img = batch['rgbs'][0]
        #gt_pose = batch['gt_poses'][0]

        if self.hparams.lamda > 0:
            masks = batch['masks'][0]
            depths = batch['depths'][0]
        
        if mode=="f_to_g":
            c2w = torch.mm(batch['poses'][0],torch.inverse(self.relative_pose(0)))
        else:
            c2w = self.model_pose(idx)

        # Calculating pose loss metric - pose loss, quartenian distances, geodesical distances
        pose_loss = None
        if mode=="f_to_g":
            with torch.no_grad():
                pose_loss, _ = pose_metrics(c2w, self.model_pose(idx))

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
            return pose_loss, loss, loss_rgb, loss_depth, psnr_
        else:
            loss = loss_rgb
            return pose_loss, loss, loss_rgb, 0, psnr_


    def training_step(self, batch, batch_nb):

        if self.relative_pose_optimization:
            _, loss,loss_rgb,loss_depth,psnr_ = self.common_training_step(batch['data_g'], mode = None)
        else:
            _, loss,loss_rgb,loss_depth,psnr_ = self.common_training_step(batch, mode = None)

        if self.relative_pose_optimization:
            assert batch['data_g']["pose_index"][0] == batch['data_f']["pose_index"][0]
            pose_loss_f, loss_f, loss_rgb_f, loss_depth_f, psnr_f = self.common_training_step(batch['data_f'], mode = "f_to_g")
            loss, loss_rgb, loss_depth =  loss + loss_f, loss_rgb + loss_rgb_f, loss_depth + loss_depth_f
        
        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        self.log('train/rgb_loss', loss_rgb)
        self.log('train/psnr', psnr_, prog_bar=True)

        if self.relative_pose_optimization:
            self.log('train/pose_loss',pose_loss_f)
            #self.log('train/quat_distance', quat_distance)
        if self.hparams.lamda > 0:
            self.log('train/depth_loss', loss_depth)

        return loss

    def validation_step(self, batch, batch_nb):

        idx = batch['idx'].detach().cpu().numpy()[0]
        img = batch['rgbs'][0]
        # gt_pose = batch['gt_poses'][0]

        if self.hparams.lamda > 0:
            masks = batch['masks'][0]
            depths = batch['depths'][0]

        c2w = self.model_pose(idx, stage="val")

        # with torch.no_grad():
        #     pose_loss, quat_distance = pose_metrics(c2w, gt_pose)

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
        # log['pose_loss'] = pose_loss
        # log['quat_distance'] = quat_distance

        return log

    def validation_epoch_end(self, outputs):

        mean_rgb_loss = torch.stack([x['val_rgb_loss'] for x in outputs]).mean()
        if self.hparams.lamda > 0:
            mean_depth_loss = torch.stack([x['val_depth_loss'] for x in outputs]).mean()
        
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        # mean_pose_loss = torch.stack([x['pose_loss'] for x in outputs]).mean()
        # mean_quat_distance = torch.stack([x['quat_distance'] for x in outputs]).mean()

        self.log('val/rgb_loss', mean_rgb_loss)
        if self.hparams.lamda > 0:
            self.log('val/depth_loss', mean_depth_loss)

        self.log('val/psnr', mean_psnr, prog_bar=True)
        # self.log('val/pose_loss', mean_pose_loss, prog_bar=False)
        # self.log('val/quat_distance', mean_quat_distance, prog_bar=False)

def get_trainer(dir_name, exp_name, hparams, max_epochs, min_delta = 0.5, resume_from_checkpoint = None):

    checkpoint_callback = ModelCheckpoint(
                                filepath=os.path.join(f'inc_ckpts/{dir_name}/{exp_name}','{epoch:d}'), 
                                monitor='val/psnr',
                                mode='max',
                                save_top_k=5
                        )
    early_stop_callback = EarlyStopping(
                            monitor='val/psnr',
                            min_delta=min_delta,
                            patience=3,
                            verbose=False,
                            mode='max'
                        )
    
    if resume_from_checkpoint is None:
        logger = TestTubeLogger(
                    save_dir="inc_logs/{}".format(dir_name),
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
        keyframe_indexes = [hyp.start_g]
        # (Initialise training from scractch)
        trainer = get_trainer(dir_name = hyp.exp_name,
                              exp_name = "nerf_{}".format(hyp.start_g), 
                              hparams = hparams, min_delta = 0.0, max_epochs = 2000)
        poses = torch.FloatTensor([[ 1.,  0.,  0.,  0.], [ 0., -1.,  0.,  0.], [ 0.,  0., -1.,  0.], [ 0.,  0.,  0.,  1.]]).unsqueeze(0)
        model_nerf = NeRFSystem(poses_to_train = [hyp.start_g], poses_to_val = [hyp.start_g], 
                                inital_poses = poses, keyframe_indexes = keyframe_indexes, 
                                freeze_nerf = False, pose_optimization = False, relative_pose_optimization = False,
                                hparams=hparams)
        trainer.fit(model_nerf)
        checkpoint_nerf_ = model_nerf.state_dict()
        initial_index = 1
        save_nerf_params(0, hyp, poses, checkpoint_nerf_, keyframe_indexes)
    else:
        # (Resume training from a serialized object)
        params = torch.load(hyp.saved_params)
        initial_index = params["index"] + 1
        checkpoint_nerf_ = params["nerf_weights"]
        poses = params["poses"]
        keyframe_indexes = params["poses_trained"]
        print("Resuming incremental training from index : {}".format(initial_index))

    for i in range(initial_index, hyp.end_g - hyp.start_g):

        # add keyframe index to list of keyframes
        keyframe_indexes.append(hyp.start_g + i) # contains global indexed of all keyframes trained

        # pose optimisation first
        trainer = get_trainer(dir_name = hyp.exp_name, exp_name = "pose_opt_{}".format(i + hyp.start_g), hparams = hparams, max_epochs = 1000)
        poses = torch.cat([poses, poses[i-1].unsqueeze(0)], dim = 0)
        poses_to_train = [i + hyp.start_g]
        poses_to_val = [i + hyp.start_g]
        model_pose = NeRFSystem(
                        poses_to_train = poses_to_train, poses_to_val = poses_to_val, 
                        inital_poses = poses, keyframe_indexes = keyframe_indexes, 
                        freeze_nerf = True, pose_optimization = True, relative_pose_optimization = True,
                        hparams=hparams
                    )
        load_ckpt(model_pose, checkpoint_nerf_, 'nerf_coarse')
        load_ckpt(model_pose, checkpoint_nerf_, 'nerf_fine')
        trainer.fit(model_pose)
        checkpoint_pose_ = model_pose.state_dict()

        # fine-tune the nerf after pose optimization
        trainer = get_trainer(dir_name = hyp.exp_name, exp_name = "nerf_{}".format(i + hyp.start_g), hparams = hparams, max_epochs = 1000)
        assert len(model_pose.model_pose.state_dict()["r"]) == 1

        pose_model = LearnPose(1, learn_R=False, learn_t=False, init_c2w = poses[i-1].unsqueeze(0))
        pose_model.eval()
        load_ckpt(pose_model, checkpoint_pose_ , 'model_pose')
        
        with torch.no_grad():
            optimized_pose = pose_model(0)

        poses[i] = optimized_pose        
        poses_to_train_local = list(np.linspace(1, i, 5, dtype=int)) if i >= 5 else [index for index in range(1, i+1)]
        poses_to_train_global = [hyp.start_g + x for x in poses_to_train_local]
        poses_to_val_global = [i + hyp.start_g]
        print("Finetuning Nerf for poses : {} ".format(poses_to_train_global))

        model_nerf = NeRFSystem(
                        poses_to_train = poses_to_train_global, poses_to_val = poses_to_val, 
                        inital_poses = poses, keyframe_indexes = keyframe_indexes,
                        freeze_nerf = False, pose_optimization = True, relative_pose_optimization = True,
                        hparams=hparams
                    )
        load_ckpt(model_nerf, checkpoint_nerf_, 'nerf_coarse')
        load_ckpt(model_nerf, checkpoint_nerf_, 'nerf_fine')
        trainer.fit(model_nerf)
        checkpoint_nerf_ = model_nerf.state_dict()

        pose_model = LearnPose(len(poses_to_train_local), learn_R=False, learn_t=False, init_c2w = poses[poses_to_train_local])
        pose_model.eval()
        load_ckpt(pose_model, checkpoint_nerf_ , 'model_pose')

        with torch.no_grad():
            for ind, pose in enumerate(poses_to_train_local):
                poses[pose] = pose_model(ind)

        save_nerf_params(i + hyp.start_g, hyp, poses, checkpoint_nerf_, keyframe_indexes)

def save_nerf_params(index, hyp, poses, checkpoint_nerf_, poses_trained):
    
    params = {
            "index" : index, 
            "nerf_weights" : checkpoint_nerf_,
            "poses" : poses,
            "poses_trained" : poses_trained
        }

    if not os.path.isdir("inc_nerf_dumps/{}".format(hyp.exp_name)):
        os.mkdir("inc_nerf_dumps/{}".format(hyp.exp_name))

    torch.save(params, "inc_nerf_dumps/{}/saved_{}".format(hyp.exp_name, index))



if __name__ == '__main__':

    if hyp.DEBUG:
        from opt import get_hparams
        hparams = get_hparams("commands.txt")
    else:
        hparams = get_opts()
        
    main(hparams)
