from typing import Dict
import torch
import torch.nn.functional as F
from loguru import logger

from csbev.utils.metrics import skewed_mse_loss


class LossHelper:
    def __init__(
        self,
        cfg: Dict,
        lambdas: Dict[str, float],
        code_dim,
    ):
        self.cfg = cfg
        self.losses = cfg.keys()
        self.lambdas = lambdas
        logger.info("Loss = {}".format(" + ".join(
            [f"{self.lambdas[loss]}*{loss}" for loss in self.lambdas]
        )))
        
        self.code_dim = code_dim

    def compute_loss(self, batch, pred, zinfo, z=None, n_iters=0):
        loss = {}
        for loss_name in self.losses:
            loss_fn = getattr(self, f"_{loss_name}_loss")
            loss_val = loss_fn(
                batch=batch,
                gt=batch["y"],
                pred=pred,
                zinfo=zinfo,
                z=z,
            )
            
            delay = self.cfg.get(loss_name, {}).get("delay", 0)
            if n_iters < delay:
                loss_val *= 0.0
            loss[loss_name] = loss_val
        
        total_loss = self.compute_weighted_loss(loss)
        loss_dict = {k: v.clone().item() for k, v in loss.items()}
        
        return total_loss, loss_dict

    def _reconstruction_loss(self, gt: torch.Tensor, pred: torch.Tensor, **kwargs):
        cfg = self.cfg.get("reconstruction", {})
        if cfg.get("recons", "mse") == "mse":
            return F.mse_loss(pred, gt)
        elif cfg.recons == "skewed_mse":
            return skewed_mse_loss(
                target=gt,
                pred=pred,
                factor=cfg.get("skew_factor", 1),
                normalization=cfg.get("normalization", "standard"),
                aggregation=cfg.get("aggregation", "mean"),
            )
        elif cfg.recons == "l1":
            return F.l1_loss(pred, gt)

    def _bottleneck_loss(self, zinfo, **kwargs):
        return zinfo[0]

    def _assignment_loss(self, batch, z, **kwargs):
        if batch.get("lr_aug", False):
            z = batch["z"] #[inverse_perm]
            z = z.reshape(2, -1, *z.shape[1:])

            # only constrain the first latent group to be consistent
            z = torch.split(z, self.code_dim, dim=2)[0]
            loss = F.mse_loss(z[0], z[1])               
        else:
            return z.new_zeros(())

    def _velocity_loss(self, gt, pred, **kwargs):
        if len(pred.shape) == 3:
            pred = pred.permute(0, 2, 1)
            pred = pred.reshape(*pred.shape[:2], -1, 3)
        
        if len(gt.shape) == 3:
            gt = gt.permute(0, 2, 1)
            gt = gt.reshape(*gt.shape[:2], -1, 3)
        
        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
        
        vel_gt = ((gt[:, 1:] - gt[:, :-1])**2).sum(-1).sqrt()
        vel_pred = ((pred[:, 1:] - pred[:, :-1])**2).sum(-1).sqrt()
        loss_vel = F.mse_loss(vel_pred, vel_gt)
        return loss_vel

    def compute_weighted_loss(self, loss: Dict[str, torch.Tensor]):
        total_loss = []
        for key in self.lambdas.keys():
            lossval = loss[key]
            total_loss.append(self.lambdas[key] * lossval)

        return sum(total_loss)