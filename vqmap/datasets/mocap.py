import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import scipy.io as sio
from vqmap.datasets.utils import align_pose
from vqmap.utils.quaternion import *
from vqmap.utils.skeleton import skeleton_initialize
from tqdm import tqdm


FPS = 50
coarse_action_enumerator = {
    0: 'smallSniffZ',
    1: 'headZ',
    2: 'groomZ',
    3: 'rearedSlowZ',
    4: 'scrunchedZ',
    5: 'crouchedZ',
    6: 'crouchedFastZ',
    7: 'rearedZ',
    8: 'groundExpZ',
    9: 'exploreZ',
    10: 'stepsExploreZ',
    11: 'locZ',
    12: 'fastLocZ'
}

class Mocap(Dataset):
    """Animal motion capture dataset
    """
    def __init__(self, datapath, seqlen, kind='xyz'):
        super().__init__()
        
        self.datapath = datapath
        self.seqlen = seqlen
        self.max_len = seqlen
        
        self.sampling = 'conseq'
        self.sampling_step = 1
        
        self.kind = kind
        
        self._action_classes = coarse_action_enumerator
        self._load_data()

    def _load_data(self):
        data = np.load(self.datapath, allow_pickle=True)[()]

        if self.kind == 'xyz':
            datakind = 'joints3D'
            self.pose3d = [pose/10 for pose in data[datakind]]
        elif self.kind == 'quat':
            datakind = 'quaternion'
            self.pose3d = [pose for pose in data[datakind]]
        elif self.kind == 'cont6d':
            datakind = 'cont6d'
            self.pose3d = [pose for pose in data[datakind]]
        
        self.pose3d = [np.reshape(pose, (pose.shape[0], -1)) for pose in self.pose3d]
        
        self._num_frames = [poseseq.shape[0] for poseseq in self.pose3d]
        self.actions = data['actions']
        assert len(self.pose3d) == len(self.actions)
    
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
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, index):
        return self._get_item_data_index(index)
    

class MocapUnannotated(Dataset):
    def __init__(
        self,
        datapath, seqlen,
        stride=1,
        kind='xyz',
        downsample=1,
        scale=1,
        keep_keypoints=None,
    ):
        super().__init__()
        
        self.datapath = datapath
        self.seqlen = seqlen
        self.max_len = seqlen
        
        self._action_classes = coarse_action_enumerator
        
        self.stride = stride
        
        self.kind = kind
        assert kind in ['xyz', 'quat', 'cont6d']
        self.skeleton = skeleton_initialize()
        self.downsample = downsample
        self.scale = scale
        
        self.keep_keypoints = keep_keypoints
        
        self.randomize = False
        
        self._load_data()
        
    def _load_data(self):
        self.pose3d = []
        self.num_frames = []

        for dp in self.datapath:

            if dp.split('_')[-1] != 'AVG0.mat':
                name = dp.split('_')[-1].replace('.mat', '.npy')
                presave = os.path.join(os.path.dirname(dp), f'aligned_{self.kind}_{name}')
            else:
                presave = os.path.join(os.path.dirname(dp), f'aligned_{self.kind}.npy')
            if os.path.exists(presave):
                data = np.load(presave)
                # num_frames = data.shape[0]
                # max_chunks = num_frames // self.seqlen
                # keep_len = max_chunks * self.seqlen
                # data = data[:keep_len]
                print(data.shape)
                self.num_frames.append(data.shape[0])
                self.pose3d.append(data)
                continue

            data = sio.loadmat(dp)["pred"]
            data = np.transpose(data, (0, 2, 1))
            
            # downsampling
            data = data[::self.downsample] * self.scale
            
            # keep max frames
            num_frames = data.shape[0]
            max_chunks = num_frames // self.seqlen
            keep_len = max_chunks * self.seqlen - self.stride
            # max_chunks = int(np.floor(num_frames / self.seqlen))
            # keep_len = max_chunks * self.seqlen
            
            data = data[:keep_len]

            self.num_frames.append(keep_len)
            print("{}: {}".format(dp, data.shape))
            
            print("Aligning ...")
            print(data.shape)
            data = self._align_data(data)
            
            if self.kind != 'xyz':
                print("Converting to rotations ...")
                quat, cont6d = self._convert_data(data)
                data = quat if self.kind == 'quat' else cont6d
            
            np.save(presave, data)
            
            # keep subsets of keypoints, if needed
            if self.keep_keypoints is not None:
                data = data[:, self.keep_keypoints]
            
            self.pose3d.append(data)
            # assume that loaded data always in xyz
        
        self.pose3d = np.concatenate(self.pose3d)
        print(f"Dataset: {self.pose3d.shape}")
    
    def __len__(self):
        # return (sum(self.num_frames) - self.stride) // self.stride + 1
        return (sum(self.num_frames) - self.seqlen) // self.stride
    
    def __getitem__(self, index):
        motion = self.pose3d[index*self.stride:(index*self.stride+self.seqlen)]
        motion = torch.from_numpy(motion).float()
        if self.randomize:
            motion = motion[:, list(np.random.permutation(motion.shape[1])), :]    
        
        motion = motion.reshape((motion.shape[0], -1))
        
        return motion, 0
    
    def _convert_data(self, joints3D):
        cont6d_joints3D = []
        quat_joints3D = []
        for seq in tqdm(joints3D.reshape(-1, self.seqlen, *joints3D.shape[1:])):
            quat_params = self.skeleton.inverse_kinematics_animal_np(seq)
            cont6d = quaternion_to_cont6d_np(quat_params)
            
            quat_joints3D.append(quat_params)
            cont6d_joints3D.append(cont6d)
        
        quat_joints3D, cont6d_joints3D = np.concatenate(quat_joints3D, 0), np.concatenate(cont6d_joints3D, 0)
        return quat_joints3D, cont6d_joints3D

    def _align_data(self, joints3D):
        aligned = []
        for seq in tqdm(joints3D.reshape(-1, self.stride, *joints3D.shape[1:])):
            aligned.append(align_pose(torch.from_numpy(seq).float())[0].numpy())
            
        aligned = np.concatenate(aligned, axis=0)
        return aligned