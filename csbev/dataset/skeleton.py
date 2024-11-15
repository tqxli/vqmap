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
        self.anterior_keypoint = anterior_keypoint
        self.posterior_keypoint = posterior_keypoint

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
        connectivity = []
        for chain in self.kinematic_tree:
            for i in range(len(chain) - 1):
                s = self.keypoints.index(chain[i])
                e = self.keypoints.index(chain[i + 1])
                connectivity.append([min(s, e), max(s, e)])
        self.connectivity = np.unique(connectivity, axis=0)

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

        # determine the facing direction
        anterior = poses[:, self.anterior_idx].mean(axis=1, keepdims=True)
        posterior = poses[:, self.posterior_idx].mean(axis=1, keepdims=True)
        forward = anterior - posterior

        if alignment == "lock_facing" and poses.shape[2] == 3:
            forward[:, :, 2] = 0
        forward = unit_vector(forward)

        # by default, align heading to the +x axis
        target = np.zeros_like(forward)
        target[:, :, 0] = 1

        # rotation matrices
        rotmat = np.stack(
            [
                rotation_matrix_from_vectors(vec1, vec2)
                for vec1, vec2 in zip(forward, target)
            ],
            axis=0,
        )
        poses_rot = rotmat @ poses.transpose((0, 2, 1))
        poses_rot = poses_rot.transpose((0, 2, 1))

        return poses_rot

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
