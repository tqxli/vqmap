from __future__ import annotations
from typing import Any, Dict, Literal
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from loguru import logger
from tqdm import tqdm
import wandb

from csbev.dataset.skeleton import SkeletonProfile
from csbev.utils.run import count_parameters, move_data_to_device


class BaseTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        expdir: str,
        skeletons: Dict[str, SkeletonProfile],
        dataset_scales: Dict[str, float],
        model: nn.Module,
        optimizer_fn: Any,
        criterion: nn.Module | None = None,
        lr_scheduler_fn: Any | None = None,
        tb_logger: Any | None = None,
        wandb_logger: Any | None = None,
        device: str = "cuda:0",
    ):
        self.cfg = cfg
        self.expdir = expdir
        self.n_epochs = cfg.train.n_epochs

        # for visualization on the fly
        self.skeletons = skeletons
        self.dataset_scales = dataset_scales
        
        # data augmentation
        self.cfg_aug = cfg.train.get("augmentation", None)

        self.model = model
        self.optimizer = self.set_optimizer(model, optimizer_fn)
        self.criterion = criterion
        self.lr_scheduler = self.set_lr_scheduler(lr_scheduler_fn)

        self.tb_logger = tb_logger
        self.wandb_logger = wandb_logger
        self.device = device
        self.model.to(self.device)
        
        self.n_iters = 0
        self.cur_epochs = 0
        
        # alternating optimization
        self.alternating_optim = cfg.train.get("alternating_optim", None)
        if self.alternating_optim is not None:
            start_epoch = self.alternating_optim.get("start_epoch", 0)
            episode = self.alternating_optim.get("episode", 25)
            self.on_epochs = list(np.arange(start_epoch, self.n_epochs, episode))
        else:
            self.on_epochs = np.arange(self.n_epochs)

    def set_optimizer(self, model, optimizer_fn):
        model_parameters = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Total trainable parameters: {count_parameters(model)}M")
        return optimizer_fn(params=model_parameters)

    def set_lr_scheduler(self, lr_scheduler_fn):
        if lr_scheduler_fn is not None:
            return lr_scheduler_fn(optimizer=self.optimizer)
        return None

    def train(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 50,
        val_freq: int = 1,
        save_freq: int = -1,
        monitor: str = "val_loss",
    ):
        best_score = np.inf
        for cur_epoch in range(n_epochs):
            metadata = {
                "epoch": cur_epoch,
                "lr": [group["lr"] for group in self.optimizer.param_groups],
            }
            self.train_epoch(train_loader, cur_epoch)

            if cur_epoch % val_freq == 0:
                val_loss = self.val_epoch(val_loader, cur_epoch)
                metadata["val_loss"] = val_loss
                
                recon_loss = val_loss[0]
                if recon_loss < best_score:
                    best_score = recon_loss
                    self.save_model(
                        f"{self.expdir}/checkpoints/model_best.pth", metadata
                    )

            self.save_model(f"{self.expdir}/checkpoints/model.pth", metadata)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            lr = self.optimizer.param_groups[0]["lr"]
            self.tb_logger.add_scalar("lr", lr, cur_epoch + 1)

    def save_model(self, save_to, metadata=None):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.cfg,
            "metadata": metadata,
        }
        torch.save(state_dict, save_to)

    def train_step(self, batch):
        out, loss, loss_dict = self.model(batch)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        
        self.n_iters += 1
        self.model.cur_iters = self.n_iters
        
        self.cur_epochs += 1
        self.model.cur_epochss = self.cur_epochs

        return out, loss, loss_dict

    def val_step(self, batch):
        with torch.no_grad():
            out, loss, loss_dict = self.model(batch)
        return out, loss, loss_dict

    def train_epoch(
        self, train_loader, cur_epoch: int,
    ):
        self.model.train()
        
        tot_losses = 0
        train_losses = 0
        total_num_samples = 0

        for idx, batch in tqdm(enumerate(train_loader)):
            batch = self.batch_preproc(batch)
            batch = move_data_to_device(batch, self.device)

            out, loss, loss_dict = self.train_step(batch)

            tot_losses += loss.item()
            train_losses += np.array([v for k, v in loss_dict.items()])
            total_num_samples += 1

        # log losses
        msg, losses_ind, loss_total = self.log_results(
            cur_epoch,
            total_num_samples,
            tot_losses,
            train_losses,
            loss_dict,
            stage="train",
        )

    def val_epoch(
        self, val_loader, cur_epoch: int,
    ):
        self.model.eval()

        tot_losses = 0
        valid_losses = 0
        total_num_samples = 0

        for idx, batch in enumerate(val_loader):
            batch = self.batch_preproc(batch)
            batch = move_data_to_device(batch, self.device)

            out, loss, loss_dict = self.val_step(batch)

            tot_losses += loss.item()
            train_losses += np.array([v for k, v in loss_dict.items()])
            total_num_samples += 1

        # log losses
        msg, losses_ind, loss_total = self.log_results(
            cur_epoch,
            total_num_samples,
            tot_losses,
            valid_losses,
            loss_dict,
            stage="val",
        )
        return loss_total

    def log_results(
        self, cur_epoch, total_num_samples, tot_losses, losses, loss_dict, stage="train"
    ):
        loss_names = list(loss_dict.keys())
        losses /= total_num_samples
        losses_str = " ".join(
            "{}: {:.4f}".format(x, y) for x, y in zip(loss_names, losses)
        )

        tot_losses /= total_num_samples

        msg = (
            f"Epoch[{cur_epoch+1}/{self.n_epochs}]"
            + " Total {} loss: {:.4f} | {}".format(stage, tot_losses, losses_str)
        )

        logger.info(msg)
        for name, loss in zip(loss_names, losses):
            self.tb_logger.add_scalar(f"{stage}/{name}", loss, cur_epoch + 1)

            if self.wandb_logger is not None:
                self.wandb_logger.log({f"{stage}/{name}": loss}, step=cur_epoch + 1)

        return msg, losses, tot_losses

    def log_metrics(self, cur_epoch, total_num_samples, metrics, metric_names, stage="train"):
        metrics /= total_num_samples
        for name, metric in zip(metric_names, metrics):
            self.tb_logger.add_scalar(f"{stage}/{name}", metric, cur_epoch + 1)

    def batch_preproc(self, batch: Dict[str, Any], stage: Literal["train", "val"] = "train") -> Dict[str, Any]:
        if "Y" not in batch:
            batch["y"] = batch["x"].detach().clone()
        
        return batch
