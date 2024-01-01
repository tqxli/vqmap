import torch
from torch.utils.data import Dataset
import numpy as np
import random
import scipy.io as sio
from vqmap.utils.quaternion import *
from vqmap.utils.skeleton import skeleton_initialize, skeleton_initialize_v2
from tqdm import tqdm


class MocapContBase(Dataset):
    """
    Parse continuous motion capture data
    """
    def __init__(
        self, datapath, seqlen,
        skeleton='rat23',
        data_rep='xyz', 
        stride=1, downsample=1, scale=1,
        normalize=False,
    ):
        super().__init__()
        
        self.datapath = datapath
        self.seqlen = seqlen
        self.max_len = seqlen
        
        # data representations
        self.data_rep = data_rep
        assert data_rep in ['xyz', 'quat', 'cont6d']
        self.pose_profile = skeleton_initialize_v2(skeleton)
        self.stride = stride
        self.downsample = downsample
        self.scale = scale
        self.normalize = normalize

        self._load_data()
    
    def _load_data(self):
        """ Load raw motion capture data from the List datapath
        """
        self.pose3d, self.num_frames = [], []

        for dp in self.datapath:
            data = self._load(dp)
            data = data[::self.downsample] * self.scale
            data = self._trim(data)
            data = self._align(data)
            data = self._normalize(data)
            data = self._convert(self.data_rep, data)
            
            self.pose3d.append(data)
        
        self.pose3d = torch.from_numpy(np.concatenate(self.pose3d)).flatten(1)
        self.pose_dim = self.pose3d.shape[-1]

        print(f"Dataset size: {self.pose3d.shape}")
    
    def __len__(self):
        return (sum(self.num_frames) - self.seqlen) // self.stride + 1
    
    def __getitem__(self, index):
        motion = self.pose3d[index*self.stride:index*self.stride+self.seqlen]
        
        return motion, 0
    
    def _load(self, datapath):
        """
        To be overwritten
        """
        data = sio.loadmat(datapath)["pred"]
        data = np.transpose(data, (0, 2, 1))
        assert data.shape[-1] == 3
        return data
    
    def _trim(self, data):
        num_frames = data.shape[0]
        max_chunks = num_frames // self.seqlen
        keep_len = max_chunks * self.seqlen
        
        data = data[:keep_len]
        self.num_frames.append(keep_len)
        
        return data
    
    def _normalize(self, data):
        if self.data_rep != "xyz" or not self.normalize:
            return data
        
        # normalize to [-1, 1]
        data = (data - data.min()) / (data.max() - data.min())
        data = 2*(data-0.5)
        
        return data
    
    def _convert(self, kind, joints3D):
        if kind == "xyz":
            return joints3D
        
        num_timepoints, num_joints = joints3D.shape[:2]
        cont6d_joints3D = np.zeros((num_timepoints, num_joints, 6))
        quat_joints3D = np.zeros((num_timepoints, num_joints, 4))
        for i, seq in enumerate(tqdm(joints3D.reshape(-1, self.seqlen, *joints3D.shape[1:]), desc="Conversion")):
            quat = self.pose_profile.inverse_kinematics_np(seq)
            cont6d = quaternion_to_cont6d_np(quat)
            
            quat_joints3D[i*self.seqlen:(i+1)*self.seqlen] = quat
            cont6d_joints3D[i*self.seqlen:(i+1)*self.seqlen] = cont6d
        
        data = quat_joints3D if kind == 'quat' else cont6d
        return data

    def _align(self, joints3D):
        aligned = np.zeros_like(joints3D)
        for i, seq in enumerate(tqdm(joints3D.reshape(-1, self.seqlen, *joints3D.shape[1:]), desc="Alignment")):
            aligned[i*self.seqlen:(i+1)*self.seqlen] = self.pose_profile.align_pose(seq)[0]

        return aligned


class MocapChunkBase(Dataset):
    def __init__(
        self,
        datapath, seqlen,
        kind='xyz',
        sampling="conseq", sampling_step=1,
    ):
        super().__init__()
        
        self.datapath = datapath
        self.seqlen = seqlen
        
        self.max_len = seqlen
        self.sampling = sampling
        self.sampling_step = sampling_step
        
        self.kind = kind
        self._load_data()
    
    def action_to_action_name(self, action):
        return self._action_classes[action]
    
    def _get_item_data_index(self, data_index):
        nframes = self._num_frames[data_index]

        if self.seqlen == -1 and (self.max_len == -1 or nframes <= self.max_len):
            frame_ix = np.arange(nframes)
        else:
            num_frames = self.seqlen if self.seqlen != -1 else self.max_len
            if num_frames > nframes:
                fair = False  # True
                if fair:
                    # distills redundancy everywhere
                    choices = np.random.choice(range(nframes),
                                               num_frames,
                                               replace=True)
                    frame_ix = sorted(choices)
                else:
                    # adding the last frame until done
                    ntoadd = max(0, num_frames - nframes)
                    lastframe = nframes - 1
                    padding = lastframe * np.ones(ntoadd, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, nframes),
                                               padding))

            elif self.sampling in ["conseq", "random_conseq"]:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(nframes),
                                           num_frames,
                                           replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")

        inp, action = self.get_pose_data(data_index, frame_ix)
        return inp, action
    
    def get_pose_data(self, data_index, frame_ix):
        poseseq = self.pose3d[data_index][frame_ix]
        action = self.actions[data_index]
        
        return poseseq, action