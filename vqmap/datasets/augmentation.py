import numpy as np
import torch
from copy import deepcopy


def lateral_flip(motion):
    """_
    [B, T, D=J*3] aligned pose
    """
    if isinstance(motion, torch.Tensor):
        motion = motion.clone()
    elif isinstance(motion, np.ndarray):
        motion = deepcopy(motion)

    # reshape into [B, T, J, 3]
    motion = motion.reshape(*motion.shape[:2], -1, 3)
    # assume aligned to +x
    motion[:, :, :, 1] = -motion[:, :, :, 1]
    return motion.reshape(*motion.shape[:2], -1)


def add_noise(motion, noise_type='gaussian'):
    if noise_type == 'gaussian':
        noise = np.random.randn(*motion.shape)
    else:
        noise = np.zeros_like(motion)
    return motion + noise