from __future__ import annotations
from copy import deepcopy
import os
from typing import List, Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.io as sio
from loguru import logger
from omegaconf import OmegaConf

from multiprocessing import Pool

from csbev.dataset.skeleton import SkeletonProfile
from csbev.dataset.ik import quaternion_to_cont6d_np
from csbev.utils.loader import list_files_with_exts, _deeplabcut_loader, _name_from_path


class BasePoseDataset(Dataset):
    """Base dataset class for motion capture pose data.
    """
    def __init__(
        self,
        dataroot: list[str],
        datapaths: None,
        skeleton: SkeletonProfile,
        name: str = "mocap",
        dataset_type: str = "dannce_soc",
        data_rep: Literal["xyz", "quat"] = "xyz",
        alignment: Literal["lock_facing", "lock_all", "center_only"] = "lock_facing",
        normalize: bool = False,
        scale: float = 1.0,
        seqlen: int = 128,
        t_downsample: int = 1,
        t_upsample: None | int = None,
        upsample_mode: Literal['repeat', 'interp'] = 'repeat',
        t_stride: int | None = None,
        t_jitter: int | None = None,
        torch_processing: bool = True,
        torch_bs: int = 16,
        # frame_shuffle: bool = False,
        shuffle: Literal[None, "frame", "all"] = None,
        velocity_threshold: float | None = None,
    ):
        self.dataroot = dataroot
        self.datapaths = datapaths if datapaths is not None else dataroot

        self.skeleton = skeleton
        self.dataset_type = dataset_type
        self.name = name

        self.data_rep = data_rep
        self.alignment = alignment
        self.normalize = normalize
        self.scale = scale

        self.seqlen = seqlen
        self.t_downsample = t_downsample
        self.t_stride = t_stride if t_stride is not None else seqlen
        assert self.t_stride <= seqlen and seqlen % self.t_stride == 0
        self.t_jitter = t_jitter is not None
        self.t_jitter_range = t_jitter
        
        self.t_upsample = t_upsample
        self.upsample_mode = upsample_mode
        
        # process with torch Tensors for acceleration
        self.torch_processing = torch_processing
        self.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_bs = torch_bs # actual frame number = seqlen * torch_bs
        logger.info(f"Processing with torch: {torch_processing} (device: {self.torch_device})")
        
        # only for debugging/shuffling baselines
        self.shuffle = shuffle
        self.shuffle_on = shuffle is not None
        logger.warning(f"Random shuffle: {self.shuffle}")
        
        self.velocity_threshold = velocity_threshold

        self.pose3d, self.num_frames = [], []
        self.raw_num_frames = 0

        self.data_preload()
        self._load_data()

    def data_preload(self):
        # placeholder for preloading data, not doing anything in the base class
        pass

    def _load(self, dp):
        data = sio.loadmat(dp)["pred"]
        data = np.squeeze(data)
        data = np.transpose(data, (0, 2, 1))
        assert data.shape[-1] == 3
        self.raw_num_frames += len(data)
        
        if self.torch_processing:
            data = torch.from_numpy(data).float().to(self.torch_device)
        
        return data

    def _scale(self, data: np.ndarray):
        return data[:: self.t_downsample] * self.scale

    def _upsample(self, data: np.ndarray):
        if self.t_upsample is None or self.t_upsample == 1 or self.t_downsample == 1:
            return data
        logger.info('Upsampling data by factor {} ({}).'.format(self.t_upsample, self.upsample_mode))
        
        if self.upsample_mode:
            data = torch.repeat_interleave(data, self.t_upsample, dim=0)
        else:
            data_device = data.device if isinstance(data, torch.Tensor) else None
            data = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data
            data_shape = data.shape
            data = data.reshape(data.shape[0], -1)
            data_interp = []
            for i in range(data.shape[-1]):
                data_interp.append(
                    np.interp(
                        np.arange(0, len(data), 1 / self.t_upsample),
                        np.arange(len(data)),
                        data[:, i],
                    )
                )
            data_interp = np.stack(data_interp, axis=-1)
            data_interp = data_interp.reshape(data_interp.shape[0], *data_shape[1:])
            data = torch.from_numpy(data_interp).float() if data_device is None else torch.from_numpy(data_interp).float().to(data_device)
        
        logger.info('Current data shape: {}'.format(data.shape))
        return data

    def _trim(self, data: np.ndarray):
        num_frames = data.shape[0]
        max_chunks = num_frames // self.seqlen
        keep_len = max_chunks * self.seqlen

        data = data[:keep_len]
        return data

    def _normalize(self, data: np.ndarray):
        if self.data_rep != "xyz" or not self.normalize:
            return data

        data = (data - data.min()) / (data.max() - data.min())
        data = 2 * (data - 0.5)
        return data

    def _align(self, joints3D: np.ndarray):
        if not self.alignment:
            return joints3D

        aligned_shape = (joints3D.shape[0], self.skeleton.n_keypoints, self.skeleton.datadim)
        if self.torch_processing:
            aligned = torch.zeros(aligned_shape, device=self.torch_device)
        else:
            aligned = np.zeros(aligned_shape)

        n_frames = joints3D.shape[0]
        bs = self.torch_bs if self.torch_processing else 1
        step = self.seqlen * bs
        for start in range(0, n_frames, self.seqlen * bs):
            end = start + step
            aligned[start:end] = self.skeleton.align_pose(joints3D[start:end], alignment=self.alignment)

        return aligned

    def _convert(self, kind: Literal["xyz", "angles"], joints3D: np.ndarray):
        if kind in ["xyz", "angles"]:
            return joints3D

        num_timepoints, num_joints = joints3D.shape[:2]
        cont6d_joints3D = np.zeros((num_timepoints, num_joints, 6))
        quat_joints3D = np.zeros((num_timepoints, num_joints, 4))
        for i, seq in enumerate(
            tqdm(
                joints3D.reshape(-1, self.seqlen, *joints3D.shape[1:]),
                desc="Conversion",
            )
        ):
            quat = self.skeleton.inverse_kinematics(seq)
            cont6d = quaternion_to_cont6d_np(quat)

            quat_joints3D[i * self.seqlen : (i + 1) * self.seqlen] = quat
            cont6d_joints3D[i * self.seqlen : (i + 1) * self.seqlen] = cont6d

        data = quat_joints3D if kind == "quat" else cont6d
        return data

    def _process_data(self, data: np.ndarray):
        data = self._scale(data)
        data = self._upsample(data)
        data = self._trim(data)
        data = self._align(data)
        data = self._normalize(data)
        data = self._convert(self.data_rep, data)
        return data

    def _load_single(self, dp):
        data = self._load(dp)
        data = self._process_data(data)
        n_frames = data.shape[0]
        return data, n_frames

    def _postproc_data(self):
        if self.torch_processing:
            self.pose3d = torch.concatenate(self.pose3d, dim=0).detach().cpu()
        else:
            self.pose3d = torch.from_numpy(np.concatenate(self.pose3d)).float()
        logger.info(f"Dataset {self.name}: {self.pose3d.shape}")
        self.pose_dim = self.pose3d.shape[-1]
        
        if self.shuffle_on:
            if self.shuffle == "frame":
                perm = torch.randperm(self.pose3d.shape[0])
                logger.warning("Frames randomly shuffled!")
                self.pose3d = self.pose3d[perm]
            elif self.shuffle == "all":
                n_frames, n_kpts = self.pose3d.shape[:2]
                perm = torch.randperm(n_frames*n_kpts)
                logger.warning("All keypoints randomly shuffled!")
                self.pose3d = self.pose3d.flatten(0, 1)[perm].reshape(n_frames, n_kpts, -1)
        
        if self.velocity_threshold is not None:
            # filter out frames with velocity above threshold
            pose3d = deepcopy(self.pose3d)
            pose3d = pose3d.reshape(-1, self.seqlen, *pose3d.shape[1:])
            pose3d_vel = pose3d[:, 1:, :] - pose3d[:, :-1, :]
            pose3d_vel = torch.norm(pose3d_vel, dim=-1).mean(-1)
            # print(pose3d_vel.min(), pose3d_vel.max(), torch.median(pose3d_vel))
            
            # pose3d_vel_sorted = np.sort(pose3d_vel.flatten(0, 1))
            # n_samples = pose3d_vel_sorted.shape[0]
            # print(pose3d_vel_sorted[int(n_samples * 0.95)])
            
            to_remove = []
            for seqid, seq in enumerate(pose3d_vel):
                if torch.any(seq > self.velocity_threshold):
                    to_remove.append(seqid)
            to_keep = set(range(pose3d.shape[0])) - set(to_remove)
            to_keep = torch.tensor(list(to_keep))
            logger.info('Filtered out {} sequences with velocity above threshold {}.'.format(
                len(to_remove), self.velocity_threshold))
            self.pose3d = pose3d[to_keep].flatten(0, 1)
            logger.info(f"Dataset {self.name} after velocity filtering: {self.pose3d.shape}")

    def _load_data(self):
        for dp in tqdm(self.datapaths, desc="Preprocess"):
            data, n_frames = self._load_single(dp)
            self.pose3d.append(data)
            self.num_frames.append(n_frames)

        self._postproc_data()

    def __len__(self):
        # return (sum(self.num_frames) - self.seqlen) // self.t_stride + 1
        return (len(self.pose3d) - self.seqlen) // self.t_stride + 1

    def __getitem__(self, index):
        start = index * self.t_stride
        end = start + self.seqlen

        jitter = (
            np.random.choice(
                np.arange(-self.t_jitter_range, self.t_jitter_range + 1), size=1,
            )
            if self.t_jitter
            else 0
        )

        start = max(0, start + jitter)
        end = min(len(self.pose3d), end + jitter)

        motion = self.pose3d[start:end]

        be = self.skeleton.body_region_indices

        sample = {"x": motion, "be": torch.from_numpy(be).long()}

        return sample


class SimpleCompiledPoseDataset(BasePoseDataset):
    """A single data file containing all mocap data as a Dict[expname, np.ndarray].
    """
    def data_preload(self):
        self.predictions = np.load(self.dataroot, allow_pickle=True)
        if self.dataroot.endswith('.npy'):
            self.predictions = self.predictions[()]

    def _load(self, dp):
        data = self.predictions[dp]
        if self.torch_processing:
            data = torch.from_numpy(data).float().to(self.torch_device)
        self.raw_num_frames += len(data)
        return data


    def _load_data(self):
        predictions = np.load(self.dataroot, allow_pickle=True)[()]
        self.pose3d, self.num_frames = [], []
        self.raw_num_frames = 0

        for dp in tqdm(self.datapaths, desc="Preprocess"):
            data = predictions[dp]
            self.raw_num_frames += len(data)
            data = self._process_data(data)
            self.pose3d.append(data)
class SyntheticPoseDataset(SimpleCompiledPoseDataset):
    """Generate synthetic dataset with augmented skeleton schemes,
    which would not be easily resolved by switching the skeleton file in `configs/skeleton`.
    """
    def __init__(
        self,
        skeleton_augmentation: Literal[None, "avg", "subset"] = None,
        skeleton_subset: List[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.skeleton_augmentation = skeleton_augmentation
        self.skeleton_subset = skeleton_subset
        if self.skeleton_augmentation == "subset":
            assert skeleton_subset is not None, "Applying subset augmentation requires skeleton_subset to be specified during initialization."
            self.skeleton_subset = OmegaConf.to_object(skeleton_subset)
        
        self._postprocess()
    
    def _postprocess(self):
        if self.skeleton_augmentation == "avg":
            connectivity = self.skeleton.connectivity
            logger.info(f"Skeleton connectivity: {connectivity.shape}")
            self.debug_vis_pose3d = deepcopy(self.pose3d[0])
            self.pose3d = self.pose3d[:, connectivity, :].mean(dim=2) # average connected kpt pairs
            
            skeleton = deepcopy(self.skeleton)
            skeleton._transform_skeleton(scheme=self.skeleton_augmentation)
            self.skeleton_original = self.skeleton
            self.skeleton = skeleton

        elif self.skeleton_augmentation == "subset":
            skeleton = deepcopy(self.skeleton)
            skeleton._transform_skeleton(scheme=self.skeleton_augmentation, subset=self.skeleton_subset)
            logger.info(f"Skeleton subset: {self.skeleton_subset}")
            self.skeleton_original = self.skeleton
            self.skeleton = skeleton
            
            self.debug_vis_pose3d = deepcopy(self.pose3d[0])
            self.pose3d = self.pose3d[:, self.skeleton.keep_indices, :]
        # else:
        #     raise NotImplementedError
    
        logger.info(f"Augmented {self.skeleton_augmentation} dataset {self.name}: {self.pose3d.shape}")

    def _vis_augmentation(self):
        from csbev.utils.visualization import visualize_pose
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(10, 5))
        axes = []
        for i in range(2):
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            axes.append(ax)
        pose3d = deepcopy(self.pose3d[0])
        for i, (data, title, skeleton) in enumerate(zip([self.debug_vis_pose3d, pose3d], ["Original", "Augmented"], [self.skeleton_original, self.skeleton])):
            axes[i].set_title(title)
            visualize_pose(
                pose=data,
                skeleton=skeleton,
                ax=axes[i],
            )
        plt.show()


class Rat7MDataset(BasePoseDataset):
    def _load(self, dp):
        data = sio.loadmat(dp)
        data = data["mocap"][0][0]
        data = np.stack(data, axis=1)
        assert data.shape[-1] == 3
        self.raw_num_frames += len(data)

        if self.torch_processing:
            data = torch.from_numpy(data).float().to(self.torch_device)
        return data

    def _interp_nan(self, data: np.ndarray):
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data = data.detach().cpu().numpy()
        
        nans = np.isnan(data)
        fcn = lambda z: z.nonzero()[0]
        for joint_idx in range(data.shape[1]):
            for coord_idx in range(data.shape[2]):
                data[nans[:, joint_idx, coord_idx], joint_idx, coord_idx] = np.interp(
                    fcn(nans[:, joint_idx, coord_idx]),
                    fcn(~nans[:, joint_idx, coord_idx]),
                    data[~nans[:, joint_idx, coord_idx], joint_idx, coord_idx],
                )
        if is_tensor:
            data = torch.from_numpy(data).float().to(self.torch_device)
        
        return data

    def _process_data(self, data: np.ndarray):
        data = self._scale(data)
        data = self._trim(data)
        data = self._interp_nan(data)
        data = self._align(data)
        data = self._normalize(data)
        data = self._convert(self.data_rep, data)
        return data


class Topdown2DPoseDataset(SimpleCompiledPoseDataset):
    """Synthesize 2D pose data from an existing 3D pose dataset.
    """
    def __init__(self, keypoints_remove: List[str], keep_z: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.keypoints_remove = keypoints_remove
        keypoints_keep = [k for k in self.skeleton.keypoints if k not in self.keypoints_remove]
        self.keypoints_keep_indices = np.array([self.skeleton.keypoints.index(k) for k in keypoints_keep])
        self.skeleton.n_keypoints = len(keypoints_keep)
        
        self.keep_z = keep_z
        self.skeleton.datadim = 2 if not self.keep_z else 3
        self.skeleton.keypoints = [self.skeleton.keypoints[idx] for idx in self.keypoints_keep_indices]
        logger.warning(f"Topdown2DPoseDataset: skeleton keypoints n={len(self.skeleton.keypoints)}")
        self.skeleton.skeleton_name += "_2d"

        body_region_indices = []
        for keypoint in self.skeleton.keypoints:
            for region, kpts in self.skeleton.body_mapping.items():
                if keypoint in kpts:
                    body_region_indices.append(self.skeleton.body_regions.index(region))
        self.skeleton.body_region_indices = np.array(body_region_indices)
        
        # fix kinematic tree for visualization
        kinematic_tree = self.skeleton.kinematic_tree
        if kinematic_tree is not None:
            for idx, chain in enumerate(kinematic_tree):
                kinematic_tree[idx] = [k for k in chain if k not in self.keypoints_remove]
            kinematic_tree_indices = [
                [self.skeleton.keypoints.index(kpt) for kpt in chain] for chain in kinematic_tree
            ]
        self.skeleton.kinematic_tree = kinematic_tree
        self.skeleton.kinematic_tree_indices = kinematic_tree_indices   

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch["x"] = batch["x"][:, self.keypoints_keep_indices]
        if not self.keep_z:
            batch["x"] = batch["x"][:, :, :2]
        else:
            batch["x"][:, :, 2] = 0
        batch["be"] = batch["be"]
        return batch


class CalMS21PoseDataset(SimpleCompiledPoseDataset):
    def data_preload(self):
        self.predictions = np.load(self.dataroot, allow_pickle=True)[()]['annotator-id_0']
        self.datapaths = sorted(list(self.predictions.keys()))
    
    def _process_data(self, data: np.ndarray):
        data = self._scale(data)
        data = self._trim(data)
        # bugfix: transpose first, otherwise the two animals are mixed in time
        if isinstance(data, torch.Tensor):
            data = data.permute(1, 0, 2, 3)
            data = data.flatten(0, 1)
            data = data.permute(0, 2, 1)
        else:
            data = data.transpose((1, 0, 2, 3))
            data = data.reshape(-1, *data.shape[2:])
            data = data.transpose((0, 2, 1))
        data = self._align(data)
        data = self._normalize(data)
        return data

    def _load(self, dp):
        data = self.predictions[dp]["keypoints"] # [n_frames, 2-mice, 2, 7]
        if self.torch_processing:
            data = torch.from_numpy(data).float().to(self.torch_device)
        return data

    def _load_data(self):
        for dp in tqdm(self.datapaths, desc="Preprocess"):
            data, n_frames = self._load_single(dp)
            if self.check_nan(data):
                logger.warning(f"NaN values found in {dp}, skipping...")
                continue
            
            self.pose3d.append(data)
            self.num_frames.append(n_frames)
        
        self._postproc_data()

    def check_nan(self, data):
        if isinstance(data, torch.Tensor):
            return torch.isnan(data).any().item()
        elif isinstance(data, np.ndarray):
            return np.isnan(data).any()


class MoSeqDLC2DDataset(CalMS21PoseDataset):
    def data_preload(self):
        self.datapaths = list_files_with_exts(self.dataroot, [".h5", ".hdf5"])
    
    def _load(self, dp):
        name = _name_from_path(dp, False, '-', True)
        new_coordinates, _, _ = _deeplabcut_loader(dp, name)
        data = new_coordinates[name][:, 1:]
        data = np.ascontiguousarray(data)  # ensure contiguous array for processing
        if self.torch_processing:
            data = torch.from_numpy(data).float().to(self.torch_device)
        return data

    def _process_data(self, data: np.ndarray):
        data = self._scale(data)
        data = self._trim(data)
        data = self._align(data)
        data = self._normalize(data)
        return data


class RatActionDataset(SyntheticPoseDataset):
    def _load_action_data(self):
        self.dataset_name = "rat23"
        
        datadrive = "/media/mynewdrive/datasets/dannce/social_rat/clustering/bigrun"
        if not os.path.exists(datadrive):
            datadrive = "/hpc/group/tdunn/tqxli/bigrun"
         
        self.frame_mapping_path = datadrive + '/classEmbedSave20230206.mat'
        self.action_label_path = datadrive + '/classEmbedSave20230206info.txt'
        self.action_mapper = np.load(
            datadrive + "/coarse_behavior_mapper.npy", allow_pickle=True
        )[()]
        self.coarse_actions = ['idle', 'sniff/head', 'groom', 'scrunched', 'crouched', 'reared', 'explore', 'locomotion', 'error']
        self.action_id_mapper = {action: idx for idx, action in enumerate(self.coarse_actions)}
        
        self.group = list(np.arange(70, 100))
        
        mappings = sio.loadmat(self.frame_mapping_path)

        foldernames = [
            fn[0][0].split('dannce_rig/')[-1].split('/SDANNCE')[0]
            for idx, fn in enumerate(mappings['FN']) if idx in self.group
        ]

        cluster_indexes = [
            fn[0][:, 0]
            for idx, fn in enumerate(mappings['ratCZN_lone']) if idx in self.group
        ]
        mapdict = {fn.split('/')[-1]: c for fn, c in zip(foldernames, cluster_indexes)}
        return mapdict

    def _data_preload(self):
        self.predictions = np.load(self.dataroot, allow_pickle=True)[()]

    def _load_data(self):
        action_data = self._load_action_data()
        self.actions, self.actions_full = [], []
        
        for dp in tqdm(self.datapaths, desc="Preprocess"):
            data, n_frames = self._load_single(dp)
            self.pose3d.append(data)
            self.num_frames.append(n_frames)
            
            action_labels = action_data[dp]
            action_labels = self._trim(action_labels)
            action_labels_coarse = [
                self.action_id_mapper[self.action_mapper[label]]
                for label in action_labels
            ]
            
            self.actions.append(action_labels_coarse)
            self.actions_full.append(action_labels)
        
        self._postproc_data()
        self.actions = torch.from_numpy(np.concatenate(self.actions)).long()
        self.actions_full = torch.from_numpy(np.concatenate(self.actions_full)).long()
    
    def __getitem__(self, index):
        start = index * self.t_stride
        end = start + self.seqlen

        motion = self.pose3d[start:end].flatten(1, 2)
        action = self.actions[start:end]

        be = self.skeleton.body_region_indices

        sample = {"x": motion, "be": torch.from_numpy(be).long(), "action": action, "tag_in": self.dataset_name, "tag_out": self.dataset_name}

        return sample


class MouseActionDataset(RatActionDataset):
    def _load_action_data(self):
        self.dataset_name = "mouse23"
        
        datadrive = "/media/mynewdrive/datasets/dannce/social_rat/clustering/mouse"
        if not os.path.exists(datadrive):
            datadrive = "/hpc/group/tdunn/tqxli/bigrun/mouse"
        
        self.coarse_actions = ['idle', 'sniff/head', 'groom', 'scrunched', 'crouched', 'reared', 'explore', 'locomotion', 'error']
        self.action_id_mapper = {action: idx for idx, action in enumerate(self.coarse_actions)}
        self.action_mapper = np.load(
            datadrive + "/coarse_behavior_mapper.npy", allow_pickle=True
        )[()]
        
        embedding_info_path = datadrive + '/mouse_embedding_info.mat'
        embedding_info = sio.loadmat(embedding_info_path)
        exp_names = [file[0][0].split('\\')[-1] for file in embedding_info['allMOUSEL_files']]
        frame_mappings = embedding_info['wrFINE']
        frame_mappings = np.stack([mappings[0][:, 0] for mappings in frame_mappings], axis=0)[:len(exp_names)]

        mapdict = {
            expname: embeddings
            for expname, embeddings in zip(exp_names, frame_mappings)
        }
        return mapdict


class RatActionLatentDataset(RatActionDataset):
    def __init__(self, model_root: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model_root = model_root
        self.embedding_path = os.path.join(self.model_root, "analysis/embeddings/rat23.pt")
        self.checkpoint_path = os.path.join(self.model_root, "checkpoints/model.pth")
        assert os.path.exists(self.model_root), f"Model root {self.model_root} does not exist."
        assert os.path.exists(self.embedding_path), f"Embedding file does not exist in {self.model_root}."
        assert os.path.exists(self.checkpoint_path), f"Checkpoint file does not exist in {self.model_root}."
        
        self.embeddings = torch.load(self.embedding_path)
        datapaths_all = sorted(list(np.load(self.dataroot, allow_pickle=True)[()].keys()))
        indices = torch.Tensor([datapaths_all.index(dp) for dp in self.datapaths]).long()
        self.embeddings = self.embeddings.reshape(len(datapaths_all), -1, *self.embeddings.shape[1:])
        self.embeddings = self.embeddings[indices]
        self.embeddings = self.embeddings.flatten(0,1)

        ckpt = torch.load(self.checkpoint_path)
        self.ds = ds = 2 ** ckpt["config"].model.encoder.channel_encoder.n_ds
        self.codebooks = [v for k, v in ckpt["model"].items() if "codebook" in k]

        assert self.embeddings.shape[0] * ds == self.pose3d.shape[0]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        
        start = index * self.t_stride // self.ds
        end = start + self.seqlen // self.ds
        
        codes = self.embeddings[start:end].long()
        codes = codes.unsqueeze(1).repeat(1,self.ds,1).flatten(0,1)
        latent = torch.cat([self.codebooks[0][codes[:, 0]], self.codebooks[1][codes[:, 1]]], dim=-1)
        sample["latent"] = latent
        sample["codes"] = codes
        return sample


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate

    # with initialize(config_path="../../configs", version_base=None):
    #     cfg = compose(config_name="dataset/mouse44.yaml").dataset
    #     del cfg.split
    #     # print(cfg)
    #     dataset = instantiate(cfg)
    # sample = dataset[0]
    # print(sample["x"].shape)

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="dataset/mouse23.yaml").dataset
        del cfg.split
    cfg['_target_'] = 'csbev.dataset.base.MouseActionDataset'
    cfg.datapaths = sorted(list(np.load(cfg.dataroot, allow_pickle=True)[()].keys()))
    cfg.datapaths = cfg.datapaths[:1]
    cfg = {k: v for k, v in cfg.items()}
    # cfg["model_root"] = '/home/tianqingli/dl-projects/duke-cluster/tqxli/experiments/vqmap2/rat23+mouse_demo+mouse23/ChannelInvariantEncoder_enc3_s2_hd256_oc32_depth3_dilation3_nh4_nq6_dec3_s2_hd256_od128_depth3_quantizer_cb12+12_dim32+32_reconsskewed_mse_lraug1.0loss0.001_delay1500_lraug_naive0.0_kptdrop0.2_3_lr0.0001'
    dataset = instantiate(cfg)
    sample = dataset[0]
    for k, v in sample.items():
        print(k, v.shape)
    
    # from csbev.dataset.loader import filter_by_keys
    # with initialize(config_path="../../configs", version_base=None):
    #     cfg = compose(config_name="dataset/mouse_demo_2d.yaml").dataset
    #     del cfg.split
    # cfg["_target_"] = "csbev.dataset.base.Topdown2DPoseDataset"

    # datapaths = sorted(list(np.load(cfg.dataroot, allow_pickle=True)[()].keys()))
    # test_ids = ["m6"]
    # cfg_test = deepcopy(cfg)
    # cfg_test.datapaths = [datapaths[idx] for idx in filter_by_keys(test_ids, datapaths)]
    # cfg_test = {k: v for k, v in cfg_test.items()}
  
    # keypoints_remove = [
    #     "ElbowL", "WristL",
    #     "ElbowR", "WristR",
    #     "KneeL", "AnkleL", "FootL",
    #     "KneeR", "AnkleR", "FootR",
    # ]
    # cfg_test["keypoints_remove"] = keypoints_remove
    
    # dataset = instantiate(cfg_test)
    # sample = dataset[0]
    # for k, v in sample.items():
    #     print(k, v.shape)
    
    # print(dataset.skeleton.kinematic_tree_indices)

    # with initialize(config_path="../../configs", version_base=None):
    #     cfg = compose(config_name="dataset/mouse_moseq_2d.yaml").dataset
    #     del cfg.split
    # dataset = instantiate(cfg)
    # sample = dataset[0]
    # for k, v in sample.items():
    #     print(k, v.shape)
    # print(sample['be'])