from __future__ import annotations
from typing import Dict, List, Literal
from loguru import logger
import numpy as np
import torch
from omegaconf import OmegaConf, ListConfig

from csbev.dataset.ik import (
    unit_vector,
    rotation_matrix_from_vectors,
    qbetween_np,
    qmul_np,
    qinv_np,
)


BODY_MAPPING_DEFAULT = {
    "Head": ["Snout", "EarL", "EarR"],
    "Trunk": ["SpineF", "SpineM", "SpineL", "TailBase"],
    "ForelimbL": ["ShoulderL", "ElbowL", "WristL", "HandL"],
    "ForelimbR": ["ShoulderR", "ElbowR", "WristR", "HandR"],
    "HindlimbL": ["HipL", "KneeL", "AnkleL", "FootL"],
    "HindlimbR": ["HipR", "KneeR", "AnkleR", "FootR"],
}


class SkeletonProfile:
    def __init__(
        self,
        skeleton_name: str = "rat23",
        keypoints_init: List[str] = [
            "Snout",
            "EarL",
            "EarR",
            "SpineF",
            "SpineM",
            "SpineL",
            "TailBase",
            "ShoulderL",
            "ElbowL",
            "WristL",
            "HandL",
            "ShoulderR",
            "ElbowR",
            "WristR",
            "HandR",
            "HipL",
            "KneeL",
            "AnkleL",
            "FootL",
            "HipR",
            "KneeR",
            "AnkleR",
            "FootR",
        ],
        body_mapping: Dict[str, List[str]] = BODY_MAPPING_DEFAULT,
        datadim: int = 3,
        keypoints: List[str]
        | None = [
            "SpineM",
            "SpineF",
            "Snout",
            "EarL",
            "EarR",
            "SpineL",
            "TailBase",
            "ShoulderL",
            "ElbowL",
            "WristL",
            "HandL",
            "ShoulderR",
            "ElbowR",
            "WristR",
            "HandR",
            "HipL",
            "KneeL",
            "AnkleL",
            "FootL",
            "HipR",
            "KneeR",
            "AnkleR",
            "FootR",
        ],
        kinematic_tree: List[List[str]] | None = [],
        center_keypoint: str = "SpineM",
        anterior_keypoint: str | list[str] = "SpineF",
        posterior_keypoint: str | list[str] = "SpineM",
        align_direction: Literal["x", "y", "z"] = "x",
    ):
        self.skeleton_name = skeleton_name
        self.datadim = datadim  # 2D or 3D pose data

        # reorder the keypoints if specified
        self.keypoints_init = keypoints_init
        if keypoints is None:
            self.keypoints = keypoints_init
            self.reorder_indices = np.arange(len(keypoints_init))
        else:
            self.keypoints = keypoints
            self.reorder_indices = np.array(
                [keypoints_init.index(k) for k in keypoints]
            )
        self.n_keypoints = len(self.keypoints)
        self.keypoints = OmegaConf.to_object(self.keypoints)

        # map keypoints to the corresponding body regions
        self.body_mapping = body_mapping
        self.body_regions = list(body_mapping.keys())
        self.body_region_indices = []
        for keypoint in self.keypoints:
            for region, kpts in body_mapping.items():
                if keypoint in kpts:
                    self.body_region_indices.append(self.body_regions.index(region))
        self.body_region_indices = np.array(self.body_region_indices)

        # basic preprocessing to the pose data
        self.center_keypoint = center_keypoint
        self.anterior_keypoint = anterior_keypoint if not isinstance(anterior_keypoint, str) else [anterior_keypoint]
        self.posterior_keypoint = posterior_keypoint if not isinstance(posterior_keypoint, str) else [posterior_keypoint]

        self.center_idx = self.keypoints.index(center_keypoint)
        self.anterior_idx = self.find_index_to_keypoint(anterior_keypoint)
        self.posterior_idx = self.find_index_to_keypoint(posterior_keypoint)
        self.align_direction = align_direction

        # kinematic chains for inverse kinematics, as well as visualization
        self.kinematic_tree = kinematic_tree
        if self.kinematic_tree is not None:
            self.kinematic_tree_indices = [
                [self.keypoints.index(kpt) for kpt in chain] for chain in kinematic_tree
            ]

        # infer all pairwise connections
        self.connectivity = self.get_connectivity_from_kinematic_tree(self.kinematic_tree)

    def get_connectivity_from_kinematic_tree(self, kinematic_tree):
        connectivity = []
        for chain in self.kinematic_tree:
            for i in range(len(chain) - 1):
                s = self.keypoints.index(chain[i])
                e = self.keypoints.index(chain[i + 1])
                pair = [min(s, e), max(s, e)]
                if pair not in connectivity:
                    connectivity.append(pair)
        return np.array(connectivity)

    def _transform_skeleton(self, scheme='avg', subset=None):
        # changes made in place, cautious
        if scheme == 'avg':
            keypoints_aug = []
            for (kpt1_idx, kpt2_idx) in self.connectivity:
                kpt1, kpt2 = self.keypoints[kpt1_idx], self.keypoints[kpt2_idx]
                new_kpt = f"{kpt1}-{kpt2}"
                keypoints_aug.append(new_kpt)
                
            kinematic_tree_aug = []
            n = 0
            for chain in self.kinematic_tree:
                chain_len = len(chain) - 1
                kinematic_tree_aug.append(keypoints_aug[n:n + chain_len])
                n += chain_len

            body_region_indices = np.max(self.body_region_indices[self.connectivity], axis=-1)
            
            self.skeleton_name = f"{self.skeleton_name}_{scheme}"
            self.keypoints = keypoints_aug
            self.n_keypoints = len(self.keypoints)
            self.kinematic_tree = kinematic_tree_aug
            self.kinematic_tree_indices = [
                [self.keypoints.index(kpt) for kpt in chain] for chain in kinematic_tree_aug
            ]
            self.body_region_indices = body_region_indices
            self.connectivity = self.get_connectivity_from_kinematic_tree(kinematic_tree_aug)
            
            anterior_keypoints = []
            posterior_keypoints = []
            for kpt in self.keypoints:
                if self.anterior_keypoint is not None and any(ant_kpt in kpt for ant_kpt in self.anterior_keypoint):
                    anterior_keypoints.append(kpt)
                if self.posterior_keypoint is not None and any(post_kpt in kpt for post_kpt in self.posterior_keypoint):
                    posterior_keypoints.append(kpt)
            self.anterior_keypoint = anterior_keypoints
            self.posterior_keypoint = posterior_keypoints
        
        elif scheme == 'subset':
            if subset is None:
                raise ValueError("Keypoint subset must be provided for subset scheme.")
            assert isinstance(subset, list), "Subset must be a list of keypoints."
            assert all(kpt in self.keypoints for kpt in subset), "All keypoints in subset must be in the original keypoints."
            self.keep_indices = keep_indices = np.array([self.keypoints.index(kpt) for kpt in subset])
            self.keypoints = [self.keypoints[i] for i in keep_indices]
            self.n_keypoints = len(self.keypoints)
            
            kinematic_tree_aug = []
            for chain in self.kinematic_tree:
                chain = [kpt for kpt in chain if kpt in subset]
                if len(chain) > 1:
                    kinematic_tree_aug.append(chain)
            self.kinematic_tree = kinematic_tree_aug
            self.kinematic_tree_indices = [
                [self.keypoints.index(kpt) for kpt in chain] for chain in kinematic_tree_aug
            ]
            self.body_region_indices = self.body_region_indices[keep_indices]
            self.connectivity = self.get_connectivity_from_kinematic_tree(kinematic_tree_aug)
            
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    def find_index_to_keypoint(self, keypoints):
        if isinstance(keypoints, str):
            return [self.keypoints.index(keypoints)]
        elif isinstance(keypoints, ListConfig):
            return [self.keypoints.index(kpt) for kpt in keypoints]

    def info(self):
        print(f"{self.skeleton_name}: {self.n_keypoints} keypoints")
        print(self.kinematic_tree_indices)

    def reorder_keypoints(self, keypoints: np.ndarray | torch.Tensor):
        return keypoints[:, self.reorder_indices]

    def center_pose(self, poses: np.ndarray | torch.Tensor):
        return poses - poses[:, self.center_idx : self.center_idx + 1]

    def align_pose(
        self,
        poses: np.ndarray | torch.Tensor,
        alignment: Literal["lock_facing", "lock_all", "center_only"] = "lock_facing",
    ):
        """
        Args:
            poses (np.ndarray | torch.Tensor): Pose data [N, n_keypoints, n_dim]
        """
        poses = self.reorder_keypoints(poses)
        poses = self.center_pose(poses)
        _, _, posedim = poses.shape
        
        # Early return if only centering is needed
        if alignment == "center_only":
            return poses
        
        # Determine if we're working with torch or numpy
        is_torch = isinstance(poses, torch.Tensor)
        
        # determine the facing direction
        anterior = poses[:, self.anterior_idx].mean(axis=1, keepdims=True)
        posterior = poses[:, self.posterior_idx].mean(axis=1, keepdims=True)
        forward = anterior - posterior

        if alignment == "lock_facing" and posedim == 3:
            forward[:, :, 2] = 0
        
        # Vectorized unit vector calculation
        if is_torch:
            forward_norm = torch.norm(forward, dim=2, keepdim=True)
            forward = forward / (forward_norm + 1e-8)  # Add epsilon to avoid division by zero
            
            # Create target vectors more efficiently
            target = torch.zeros_like(forward)
            target[:, :, 0] = 1
            
            # Vectorized rotation - specific optimization for PyTorch
            if poses.is_cuda:
                # Use a custom vectorized rotation function optimized for GPU
                poses_rot = self._torch_vectorized_rotation(forward, target, poses)
            else:
                # Use the existing approach for CPU
                rotmat = torch.stack(
                    [rotation_matrix_from_vectors(vec1, vec2) 
                     for vec1, vec2 in zip(forward, target)],
                    dim=0,
                )
                poses_rot = torch.bmm(rotmat, poses.transpose(1, 2)).transpose(1, 2)
        else:
            # NumPy path
            forward_norm = np.linalg.norm(forward, axis=2, keepdims=True)
            forward = forward / (forward_norm + 1e-8)
            
            # Create target vectors
            target = np.zeros_like(forward)
            target[:, :, 0] = 1
            
            # Use optimized batch rotation computation if available
            batch_size = poses.shape[0]
            if batch_size > 10:  # For small batches, the overhead might not be worth it
                # Preallocate the rotation matrices array
                rotmat = np.empty((batch_size, 3, 3))
                
                # Fill it efficiently - could be parallelized with numba if needed
                for i, (vec1, vec2) in enumerate(zip(forward.reshape(-1, 3), target.reshape(-1, 3))):
                    rotmat[i] = rotation_matrix_from_vectors(vec1, vec2)
                    
                # Batch matrix multiplication
                poses_rot = np.matmul(rotmat, poses.transpose(0, 2, 1))
                poses_rot = poses_rot.transpose(0, 2, 1)
            else:
                # Original approach for small batches
                rotmat = np.stack(
                    [rotation_matrix_from_vectors(vec1, vec2)
                     for vec1, vec2 in zip(forward.reshape(-1, 3), target.reshape(-1, 3))],
                    axis=0,
                )
                poses_rot = np.matmul(rotmat, poses.transpose(0, 2, 1))
                poses_rot = poses_rot.transpose(0, 2, 1)

        return poses_rot

    def _torch_vectorized_rotation(self, forward, target, poses):
        """Optimized rotation computation for PyTorch tensors on GPU"""
        # This implementation would depend on the specifics of rotation_matrix_from_vectors
        # But could be implemented using batch operations in PyTorch
        
        # Example implementation (would need to be adapted to match rotation_matrix_from_vectors):
        batch_size = poses.shape[0]
        forward = forward.reshape(batch_size, 3)
        target = target.reshape(batch_size, 3)
        
        # Compute rotation matrices in a vectorized way
        v = torch.cross(forward, target, dim=1)
        c = torch.sum(forward * target, dim=1).unsqueeze(1).unsqueeze(2)
        
        # Create skew-symmetric matrices
        v_x = torch.zeros(batch_size, 3, 3, device=poses.device)
        v_x[:, 0, 1] = -v[:, 2]
        v_x[:, 0, 2] = v[:, 1]
        v_x[:, 1, 0] = v[:, 2]
        v_x[:, 1, 2] = -v[:, 0]
        v_x[:, 2, 0] = -v[:, 1]
        v_x[:, 2, 1] = v[:, 0]
        
        # Rodrigues' rotation formula
        I = torch.eye(3, device=poses.device).unsqueeze(0).repeat(batch_size, 1, 1)
        rotmat = I + v_x + torch.bmm(v_x, v_x) / (1 + c)
        
        # Apply rotation
        return torch.bmm(rotmat, poses.transpose(1, 2)).transpose(1, 2)

    def inverse_kinematics(self, joints: np.ndarray):
        forward = joints[:, self.anterior_idx] - joints[:, self.posterior_idx]
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

        target = np.array([[1, 0, 0]]).repeat(len(forward), axis=0)

        # compute the root rotations
        root_quat = qbetween_np(forward, target)

        # quat_params (batch_size, n_kpts, 4)
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        for chain in self.kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                u = self.offsets[chain[j + 1]][np.newaxis, ...].repeat(
                    len(joints), axis=0
                )
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]
                v = v / np.sqrt((v ** 2).sum(axis=-1))[:, np.newaxis]
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params


if __name__ == "__main__":
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("./configs/skeleton/mouse44.yaml")
    skeleton = instantiate(cfg)
    skeleton.info()
