from typing import Dict
from copy import deepcopy

import numpy as np
import torch


def lr_flip_naive(batch: Dict[str, torch.Tensor]):
    """Augmentation function that flips the skeletons along the y axis (assumes facing +x).

    Args:
        batch (Dict[str, torch.Tensor]): input data batch.

    Returns:
        Dict[str, torch.Tensor]: Augmented data batch.
    """
    batch_aug = deepcopy(batch)
    batch_aug["x"][..., 1] = -batch_aug["x"][..., 1]
    batch_aug["y"] = batch_aug["x"]
    return batch_aug


def lr_flip(batch: Dict[str, torch.Tensor]):
    """L/R flip, but keep both the original and augmented data in the batch.

    Args:
        batch (Dict[str, torch.Tensor]): input data batch.

    Returns:
        Dict[str, torch.Tensor]: Augmented data batch.
    """
    batch_aug = lr_flip_naive(batch)
    
    # concatenate the original and augmented data (2x batch size)
    for k in ["x", "be", "y"]:
        batch_aug[k] = torch.cat([batch[k].detach().clone(), batch_aug[k]], dim=0)
    batch_aug["lr_aug"] = True
    return batch_aug


def keypoints_dropout(batch: Dict[str, torch.Tensor], max_parts: int = 3):
    """Augmentation function that drops random keypoints during training.

    Args:
        batch (Dict[str, torch.Tensor]): input data batch.
        max_parts (int, optional): Maxmimal body parts to be dropped (6 in total for most body schemes). Defaults to 3.

    Returns:
        Dict[str, torch.Tensor]: Augmented data batch.
    """
    batch_aug = deepcopy(batch)
    
    # select the body parts to drop keypoints from
    be = batch_aug["be"][0]
    parts_unique = torch.unique(be)
    parts = parts_unique[torch.randperm(parts_unique.size(0))][:max_parts]
    
    # drop 1 keypoint from each
    kpts_to_drop = []
    for part in parts:
        kpts = torch.where(be == part)[0]
        kpt_to_drop = kpts[torch.randint(0, kpts.size(0), (1,))].item()
        kpts_to_drop.append(kpt_to_drop)
    
    keep = torch.Tensor([i for i in range(be.size(0)) if i not in kpts_to_drop]).long()
    perm = torch.randperm(keep.size(0))
    keep = keep[perm]
    
    # mask data accordingly
    x_ori = deepcopy(batch_aug["x"])
    x = deepcopy(batch_aug["x"])
    batch_aug["x_ori"] = x_ori
    batch_aug["x"] = batch_aug["x"][:, :, keep]
    batch_aug["be"] = batch_aug["be"][:, keep]
    batch_aug["kpt_dropout"] = True
    
    # keep masked data (no cardinality changes) for visualzation only
    mask = torch.Tensor(kpts_to_drop).long()
    x[:, :, mask] = 0
    batch_aug["kpts_masked"] = x
    return batch_aug


def skeleton_scaling(batch: Dict[str, torch.Tensor], scale_factor: float = 0.2):
    """Augmentation function that scales the pose.

    Args:
        batch (Dict[str, torch.Tensor]): input data batch.
        scale_factor (float, optional): skeletons are scaled within the range [1-scale_fractor, 1+scale_factor]. Defaults to 0.2.

    Returns:
        Dict[str, torch.Tensor]: Augmented data batch.
    """
    batch_aug = deepcopy(batch)
    
    scale = np.random.uniform(1 - scale_factor, 1 + scale_factor)
    batch_aug["x"] *= scale
    return batch_aug
