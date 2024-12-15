from copy import deepcopy
import io
import random
from typing import Any, Dict, Literal
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

matplotlib.use("Agg")  # must keep, otherwise trigger bugs with matplotlib + tensorboard

from csbev.dataset.skeleton import SkeletonProfile
from csbev.trainer.base import BaseTrainer
from csbev.utils.visualization import make_pose_seq_overlay
from csbev.utils.run import move_data_to_device
from csbev.utils.metrics import compute_mpjpe
from csbev.utils.augmentation import lr_flip, keypoints_dropout, lr_flip_naive, skeleton_scaling


class TrainerMultiLoader(BaseTrainer):
    def batch_preproc(self, batch: Dict[str, Any], stage: Literal["train", "val"] = "train") -> Dict[str, Any]:
        batch = super().batch_preproc(batch, stage)

        if stage == "train" and self.cfg.train.augmentation is not None:
            batch = self.behavior_augmentation(batch)
        return batch

    def behavior_augmentation(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if random.random() < self.cfg_aug.get("lr_flip_prob", 0.0):
            return lr_flip(batch)
        
        if random.random() < self.cfg_aug.get("lr_flip_naive_prob", 0.0):
            return lr_flip_naive(batch)
        
        if random.random() < self.cfg_aug.get("kpt_dropout_prob", 0.0):
            max_parts = self.cfg.train.augmentation.get("kpt_dropout_max_parts", 3)
            return keypoints_dropout(batch, max_parts=max_parts)

        if random.random() < self.cfg_aug.get("skeleton_scaling_prob", 0.0):
            scale_factor = self.cfg.train.augmentation.get("skeleton_scaling_factor", 0.5)
            assert scale_factor > 0.0 and scale_factor < 1.0
            return skeleton_scaling(batch, scale_factor=scale_factor)

        return batch

    def compute_metrics(self, batch: Dict[str, torch.Tensor], out: torch.Tensor):
        """Additional metrics (e.g., MPJPE)

        Args:
            batch (Dict[str, torch.Tensor]): Input batch Dict['x', 'be', 'tag', 'z']
            out (torch.Tensor): Reconstruction [B, NUM_KEYPOINTS*3, T]
        """
        tag = batch["tag_in"]
        skeleton = self.skeletons[tag]
        keypoints = skeleton.keypoints
        data_scale = 1.0 / self.dataset_scales[tag]
        
        target = batch["x_ori"].clone().detach().cpu() * data_scale
        pred = out.clone().detach().cpu().permute(0, 2, 1)
        pred = pred.reshape(*pred.shape[:2], skeleton.n_keypoints, skeleton.datadim) * data_scale
        assert target.shape == pred.shape
        
        # MPJPE
        err = compute_mpjpe(pred, target, average=False) # [B, T, NUM_KEYPOINTS]
        err_per_joint = {
            keypoints[idx]: err[:, :, idx].mean().item()
            for idx in range(skeleton.n_keypoints)
        }
        err_per_region = {
            region: err[:, :, np.where(skeleton.body_region_indices==idx)[0]].mean().item()
            for idx, region in enumerate(skeleton.body_regions)
        }
        return err.mean().item(), err_per_joint, err_per_region

    def train_epoch(
        self, train_loader: Dict[str, DataLoader], cur_epoch: int,
    ):
        self.model.train()
        
        tot_losses = 0
        train_losses = 0
        train_metrics = 0
        train_metrics_per_joint = {tag: 0 for tag in train_loader.keys()}
        train_metrics_per_region = {tag: 0 for tag in train_loader.keys()}
        total_num_samples = 0

        tags = list(train_loader.keys())

        for batches in tqdm(zip(*[iter(loader) for loader in train_loader.values()])):
            for tag, batch in zip(tags, batches):
                batch = self.batch_preproc(batch, "train")
                batch["tag_in"] = batch["tag_out"] = tag
                batch = move_data_to_device(batch, self.device)

                out, loss, loss_dict = self.train_step(batch)

                tot_losses += loss.item()
                train_losses += np.array([v for k, v in loss_dict.items()])
                total_num_samples += 1
                
                mpjpe, err_per_joint, err_per_region = self.compute_metrics(batch, out)
                train_metrics += mpjpe
                train_metrics_per_joint[tag] += np.array([v for k, v in err_per_joint.items()])
                train_metrics_per_region[tag] += np.array([v for k, v in err_per_region.items()])

        # log losses
        msg, losses_ind, loss_total = self.log_results(
            cur_epoch,
            total_num_samples,
            tot_losses,
            train_losses,
            loss_dict,
            stage="train",
        )
        
        # log metrics
        for tag in tags:
            self.log_metrics(
                cur_epoch,
                total_num_samples,
                train_metrics_per_region[tag],
                [f"{tag}/{k}" for k in err_per_region.keys()],
                stage="train",
            )

    def val_epoch(
        self, val_loader: Dict[str, DataLoader], cur_epoch: int,
    ):
        self.model.eval()

        tot_losses = 0
        valid_losses = 0
        valid_metrics = 0
        valid_metrics_per_joint = {tag: 0 for tag in val_loader.keys()}
        valid_metrics_per_region = {tag: 0 for tag in val_loader.keys()}
        total_num_samples = 0
        n_iters = 0

        tags = list(val_loader.keys())

        for batches in tqdm(zip(*[iter(loader) for loader in val_loader.values()])):
            for tag, batch in zip(tags, batches):
                batch = self.batch_preproc(batch, "val")
                batch["tag_in"] = batch["tag_out"] = tag
                batch = move_data_to_device(batch, self.device)

                out, loss, loss_dict = self.val_step(batch)

                tot_losses += loss.item()
                valid_losses += np.array([v for k, v in loss_dict.items()])
                total_num_samples += 1
                
                mpjpe, err_per_joint, err_per_region = self.compute_metrics(batch, out)
                valid_metrics += mpjpe
                valid_metrics_per_joint[tag] += np.array([v for k, v in err_per_joint.items()])
                valid_metrics_per_region[tag] += np.array([v for k, v in err_per_region.items()])

                # visualize reconstruction
                if n_iters == 0:
                    self.visualize_reconstruction(
                        out, batch["x"], tag, cur_epoch,
                    )
            n_iters += 1

        # log losses
        msg, losses_ind, loss_total = self.log_results(
            cur_epoch,
            total_num_samples,
            tot_losses,
            valid_losses,
            loss_dict,
            stage="val",
        )
        
        # log metrics
        for tag in tags:
            self.log_metrics(
                cur_epoch,
                total_num_samples,
                valid_metrics_per_region[tag],
                [f"{tag}/{k}" for k in err_per_region.keys()],
                stage="val",
            )

        # visualize latent space
        if (
            self.cfg.train.latent_vis_freq > 0
            and cur_epoch % self.cfg.train.latent_vis_freq == 0
        ):
            for tag in val_loader.keys():
                self.visualize_latent(tag, cur_epoch)

        return losses_ind

    def visualize_reconstruction(
        self, pred: torch.Tensor, gt: torch.Tensor, tag: str, cur_epoch: int,
    ):
        fig = plt.figure(figsize=(8, 4), dpi=200)
        pred = pred.clone().detach().cpu()
        pred = pred.permute(0, 2, 1)
        pred = pred.reshape(
            *pred.shape[:2],
            self.skeletons[tag].n_keypoints,
            self.skeletons[tag].datadim,
        )
        gt = gt.detach().cpu().numpy()
        assert pred.shape == gt.shape

        for idx, poseseq in enumerate([gt, pred]):
            if self.skeletons[tag].datadim == 2:
                ax = fig.add_subplot(1, 2, idx + 1)
            elif self.skeletons[tag].datadim == 3:
                ax = fig.add_subplot(1, 2, idx + 1, projection="3d")

            ax = make_pose_seq_overlay(
                poseseq=poseseq[0],
                skeleton=self.skeletons[tag],
                n_samples=5,
                alpha_min=0.2,
                linewidth=1.0,
                marker_size=20,
                coord_limits=0.6,
                ax=ax,
                savename=None,
            )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="raw")
        buf.seek(0)

        img = np.reshape(
            np.frombuffer(buf.getvalue(), dtype=np.uint8),
            (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
        )
        img = np.transpose(img, (2, 0, 1))[:3]
        buf.close()

        self.tb_logger.add_image(f"reconstruction/{tag}", img, cur_epoch + 1)
        plt.close(fig)

        if self.wandb_logger is not None:
            self.wandb_logger.log(
                {f"reconstruction/{tag}": wandb.Image(img.transpose((1, 2, 0)))},
                step=cur_epoch + 1,
            )

    def visualize_latent(self, tag: str, cur_epoch: int):
        # sample latent codes and reconstruct trajectories
        _, traj, combos = self.model.sample_latent_codes(tag)

        # visualize
        codebook_size = self.cfg.model.bottleneck.codebook_size
        h, w = codebook_size
        fig = plt.figure(figsize=(w * 4, h * 4), dpi=100)

        for idx, combo in enumerate(combos):
            ax = fig.add_subplot(h, w, idx + 1, projection="3d")
            ax.set_title(f"Code: {combo}")
            poseseq = traj[idx].reshape(
                self.skeletons[tag].n_keypoints, self.skeletons[tag].datadim, -1
            )
            poseseq = poseseq.permute(2, 0, 1).detach().cpu().numpy()

            ax = make_pose3d_seq_overlay(
                poseseq=poseseq,
                skeleton=self.skeletons[tag],
                n_samples=5,
                alpha_min=0.2,
                linewidth=1.0,
                marker_size=20,
                coord_limits=0.6,
                ax=ax,
                savename=None,
            )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="raw")
        buf.seek(0)

        img = np.reshape(
            np.frombuffer(buf.getvalue(), dtype=np.uint8),
            (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
        )
        img = np.transpose(img, (2, 0, 1))[:3]
        buf.close()

        self.tb_logger.add_image(f"latent_space/{tag}", img, cur_epoch + 1)
        plt.close(fig)

        if self.wandb_logger is not None:
            self.wandb_logger.log(
                {f"latent_space/{tag}": wandb.Image(img.transpose((1, 2, 0)))},
                step=cur_epoch + 1,
            )
