from typing import Literal

import torch
import torch.nn.functional as F


def compute_mpjpe(pred: torch.Tensor, target: torch.Tensor, average: bool = True) -> float:
    """Compute mean per point position error.

    Args:
        pred (torch.Tensor): [B, T, NUM_KEYPOINTS, 3]
        target (torch.Tensor): [B, T, NUM_KEYPOINTS, 3]

    Returns:
        float: error values
    """
    if len(pred.shape) == 3:
        pred = pred.permute(0, 2, 1)
        pred = pred.reshape(*pred.shape[:2], -1, 3)

    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    # assert pred.shape[-1] == 3, f"Expected 3D keypoints, got {pred.shape[-1]}"
    mpjpe = ((pred - target) ** 2).sum(-1).sqrt()
    
    if average:
        return mpjpe.mean().item()
    return mpjpe


def skewed_mse_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    factor: int = 1,
    normalization: Literal["standard", "softmax"] = "standard",
    aggregation: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    if len(pred.shape) == 3:
        pred = pred.permute(0, 2, 1)
        pred = pred.reshape(*pred.shape[:2], -1, 3)
    
    if len(target.shape) == 3:
        target = target.permute(0, 2, 1)
        target = target.reshape(*target.shape[:2], -1, 3)

    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    assert pred.shape[-1] == 3, f"Expected 3D keypoints, got {pred.shape[-1]}"
    
    # compute the relative distance between center keypoint and others
    dist = ((target - target[:, :, :1])**2).sum(-1, keepdim=True).sqrt() #[B, T, NUM_KEYPOINTS, 1]
    
    dist = dist**factor + 1e-6
    
    # compute weighting factors
    if normalization == "standard":
        weights = dist / dist.sum(dim=2, keepdim=True)
    elif normalization == "softmax":
        weights = F.softmax(dist, dim=2)
    
    # compute loss and apply weights
    loss = F.mse_loss(pred, target, reduction="none")
    loss = loss * weights
    
    if aggregation == "mean":
        return loss.mean()
    elif aggregation == "sum":
        return loss.sum() / loss.shape[0]