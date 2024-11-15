from __future__ import annotations
from copy import deepcopy
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# from csbev.model.loss import LossHelper
from csbev.model.quantizer import MixBottleneckv2
from csbev.utils.metrics import skewed_mse_loss


class LossHelper(nn.Module):
    def __init__(self, cfg, lambdas: Dict[str, float], code_dim: List[int]):
        super().__init__()

        self.cfg = cfg
        self.lambdas = lambdas
        
        self.code_dim = code_dim
        self.joint_weights = None

    def compute_loss(self, batch, pred, zinfo, z=None, n_iters=0):
        target = batch["y"]  # [B, T, n_points, D]
        target = target.permute(0, 2, 3, 1).flatten(1, 2)  # [B, n_points*D, T]
        recons = self.compute_reconstruction_loss(target, pred)
        loss = {"recons": recons}

        # assignment loss
        loss = self.compute_code_assignment_loss(loss, batch, z)

        delay = self.cfg.get("assignment_delay", 0)
        if n_iters < delay:
            # print(delay, n_iters)
            loss["assignment"] *= 0.0
        
        bottleneck_loss = self.compute_bottleneck_loss(zinfo)
        
        loss_vel = self.compute_vel_loss(target, pred)
        loss["velocity"] = loss_vel
        
        loss = {**bottleneck_loss, **loss}
        total_loss = self.compute_weighted_loss(loss)
        loss_dict = {k: v.clone().item() for k, v in loss.items()}

        return total_loss, loss_dict

    def compute_reconstruction_loss(self, gt, pred):
        if self.joint_weights is None:
            if self.cfg.get("recons", "mse") == "mse":
                return F.mse_loss(pred, gt)
            elif self.cfg.recons == "skewed_mse":
                return skewed_mse_loss(
                    target=gt,
                    pred=pred,
                    factor=self.cfg.get("skew_factor", 1),
                    normalization=self.cfg.get("normalization", "standard"),
                    aggregation=self.cfg.get("aggregation", "mean"),
                )
            elif self.cfg.recons == "l1":
                return F.l1_loss(pred, gt)

        assert gt.shape == pred.shape, f"Shape mismatch: {gt.shape} vs {pred.shape}"

        gt, pred = (
            gt.reshape(*gt.shape[:2], -1, 3),
            pred.reshape(*pred.shape[:2], -1, 3),
        )
        recon_loss = F.mse_loss(pred, gt, reduction="none")        
        recon_loss = recon_loss.mean(0).mean(0).mean(-1)
        return (self.joint_weights.to(recon_loss.device) * recon_loss).mean()

    def compute_bottleneck_loss(self, zinfo):
        return zinfo[0]

    def compute_kl(self, distribution):
        # Create a centred normal distribution to compare with
        mu_ref = torch.zeros_like(distribution.loc)
        scale_ref = torch.ones_like(distribution.scale)
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl = self.kl_loss(distribution, distribution_ref)

        return kl

    def compute_code_assignment_loss(self, loss, batch, z):
        if "assignment" in self.lambdas and z is not None:
            if batch.get("lr_aug", False):
                loss["assignment"] = self.compute_assignment_loss(batch)
            else:
                loss["assignment"] = z.new_zeros(())
        return loss

    def compute_assignment_loss(self, batch):
        # B = batch["x"].shape[0]
        # inverse_perm = torch.argsort(batch["perm"])

        z = batch["z"] #[inverse_perm]
        z = z.reshape(2, -1, *z.shape[1:])

        # only constrain the first latent group to be consistent
        z = torch.split(z, self.code_dim, dim=2)[0]

        # loss = F.mse_loss(z[0], z[1], reduction="sum") / B
        loss = F.mse_loss(z[0], z[1])
        return loss

    def compute_weighted_loss(self, loss):
        total_loss = []
        for key, lossval in loss.items():
            total_loss.append(self.lambdas[key] * lossval)

        return sum(total_loss)

    def compute_vel_loss(self, gt, pred):
        if len(pred.shape) == 3:
            pred = pred.permute(0, 2, 1)
            if pred.shape[-1] % 3 != 0:
                return pred.new_zeros(())
            
            pred = pred.reshape(*pred.shape[:2], -1, 3)
        
        if len(gt.shape) == 3:
            gt = gt.permute(0, 2, 1)
            gt = gt.reshape(*gt.shape[:2], -1, 3)
        
        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
        
        vel_gt = ((gt[:, 1:] - gt[:, :-1])**2).sum(-1).sqrt()
        vel_pred = ((pred[:, 1:] - pred[:, :-1])**2).sum(-1).sqrt()
        loss_vel = F.mse_loss(vel_pred, vel_gt)
        return loss_vel


class SeqVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        bottleneck: nn.Module | None = None,
        loss_helper = None,
        lambdas: Dict[str, float] = {"recons": 1.0},
        loss_cfg: Dict = {},
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

        self.loss_helper = LossHelper(
            loss_cfg,
            lambdas,
            bottleneck.code_dim if bottleneck is not None else None,
        )
        # self.loss_helper = loss_helper(
        #     code_dim=bottleneck.code_dim if bottleneck is not None else None,
        # )
        
        self.cur_iters = 0

    def _bottleneck(self, latent: torch.Tensor):
        if self.bottleneck is None:
            return latent, {}
        
        if isinstance(self.bottleneck, MixBottleneckv2):
            return self.bottleneck(latent, self.cur_iters)
        
        return self.bottleneck(latent)

    def encode(self, batch: Dict):
        x = batch["x"]  # [B, T, n_points, D]
        x = x.permute(0, 2, 3, 1)  # [B, n_points, D, T]

        latent = self.encoder(
            x,
            pe_indices=batch["be"],
            batch_tag=batch["tag_in"],
        )  # -> [B, latent_dim, T]
        z, loss, info = self._bottleneck(latent)
        batch["z"] = z
        return z, (loss, info), latent

    def decode(self, x: torch.Tensor, batch_tag: str | None = None):
        return self.decoder(x, batch_tag=batch_tag)

    def forward(self, batch: Dict):
        if "x_ori" not in batch:
            batch["x_ori"] = deepcopy(batch["x"])
        z, z_info, latent = self.encode(batch)
        x_recon = self.decode(z, batch_tag=batch.get("tag_out", None))

        total_loss, loss_dict = self.loss_helper.compute_loss(
            batch, x_recon, z_info, z,
            n_iters=self.cur_iters,
        )

        return x_recon, total_loss, loss_dict

    def sample_latent_codes(self, tag, repeats=1):
        codevecs, combinations = self.bottleneck.get_codevecs()
        codevecs = codevecs.repeat(1, 1, repeats)
        traj = self.decode(codevecs, tag)
        return codevecs, traj, combinations

    def reconstruct(self, batch: Dict):
        z = self.encode(batch)[0]
        x_recon = self.decode(z, batch_tag=batch.get("tag_out", None))
        return x_recon


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate
    from csbev.utils.run import count_parameters

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="model/vae_ci.yaml")

    model = instantiate(cfg.model)
    print(f"Model size: {count_parameters(model):.2f}M")

    # x = torch.randn(5, 256, 20, 3)
    # batch = {"x": x, "tag": "mouse_demo"}
    # print(f"Input shape: {x.shape}")
    # x_recon, total_loss, loss_dict = model(batch)
    # print(f"Reconstruction shape: {x_recon.shape}")
    # print(f"Total loss: {total_loss} {loss_dict}")

    # codevecs, traj, _ = model.sample_latent_codes("mouse_demo")
    # print(f"Codevecs shape: {codevecs.shape}")
    # print(f"Traj shape: {traj.shape}")
