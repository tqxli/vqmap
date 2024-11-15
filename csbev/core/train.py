import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import wandb

from csbev.dataset.loader import prepare_datasets, prepare_dataloaders
from csbev.trainer.trainer import TrainerMultiLoader
from csbev.utils.run import automatic_experiment_naming, check_config, prepare_expdir


def train(cfg: DictConfig):
    # Check configuration
    # cfg = check_config(cfg)
    
    # Create experiment output directory
    expdir = prepare_expdir(cfg)

    # Specify logger (by loguru) path
    logger.add(
        os.path.join(expdir, f"stats_train.log"),
        format="{time:YYYY-MM-DD HH:mm} | {level} | {message}",
    )

    # Initialize tensorboard logger
    tb_logger = SummaryWriter(os.path.join(expdir, "tb"))

    # Fix random seed for all (torch, numpy, random)
    if cfg.get("seed", None):
        seed_everything(cfg.seed)

    # Prepare datasets and dataloaders
    datasets, skeletons = prepare_datasets(cfg)
    dataloaders = prepare_dataloaders(datasets, cfg.dataloader)

    # Save scales used in normalization for later computation of absolute errors in mm
    dataset_scales = {}
    for tag, dataset in datasets["train"].items():
        if isinstance(dataset, torch.utils.data.Subset):
            dataset_scales[tag] = dataset.dataset.scale
        else:
            dataset_scales[tag] = dataset.scale

    # Instantiate model
    model = instantiate(cfg.model)

    # Instantiate trainer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if cfg.use_wandb:
        wandb.login(key=cfg.wandb.api_key)
        wandb_logger = wandb.init(
            project=cfg.wandb.project, config=dict(cfg), name=cfg.expname,
        )
        wandb.watch(model, log="all", log_freq=1)

    trainer = TrainerMultiLoader(
        cfg=cfg,
        expdir=expdir,
        skeletons=skeletons,
        dataset_scales=dataset_scales,
        model=model,
        optimizer_fn=instantiate(cfg.optimizer),
        lr_scheduler_fn=instantiate(cfg.lr_scheduler),
        tb_logger=tb_logger,
        wandb_logger=wandb_logger if cfg.use_wandb else None,
        device=device,
    )
    OmegaConf.save(cfg, os.path.join(expdir, "config.yaml"))

    trainer.train(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        n_epochs=cfg.train.n_epochs,
        val_freq=cfg.train.val_freq,
        save_freq=cfg.train.save_freq,
        monitor=cfg.train.monitor,
    )

    if cfg.use_wandb:
        wandb.finish()


@hydra.main(config_path="../../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
