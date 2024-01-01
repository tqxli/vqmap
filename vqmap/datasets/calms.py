import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import scipy.io as sio
from vqmap.datasets.utils import unit_vector, rotation_matrix_from_vectors
from tqdm import tqdm

# 7 tracked body parts, ordered (nose, left ear, right ear, neck, left hip, right hip, tail base).
REORDER_INDICES = np.array([3, 1, 2, 0, 4, 5, 6])
# neck, le, re, nose, lh, rh, tail

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(2), (vec2 / np.linalg.norm(vec2)).reshape(2)
    x1, y1 = a
    x2, y2 = b
    c = np.dot(a, b)
    rotation_matrix = [[c, x2*y1-x1*y2], [x1*y2-x2*y1, c]]
    return rotation_matrix

def align_pose2d(sample):
    sample = sample[:, REORDER_INDICES, :]
    traj = sample[:, :1]
    sample = sample - traj
    spineline = sample[:, 3:4] - sample[:, -1:]
    spineline = unit_vector(spineline)

    x_axis = np.zeros_like(spineline)
    x_axis[:, :, 0] = 1
 
    rotmat = [rotation_matrix_from_vectors(vec1, vec2) for (vec1, vec2) in zip(spineline, x_axis)]
    rotmat = np.stack(rotmat, 0)

    sample_rot = rotmat @ sample.transpose((0, 2, 1))
    sample_rot = sample_rot.transpose((0, 2, 1))
    
    sample_rot /= 10.0
    
    return sample_rot, rotmat, traj

class CalMS21(Dataset):
    def __init__(self,
                 datapath='/media/mynewdrive/datasets/CalMS21/mab-e-baselines-master/data/calms21_task1_train.npy',
                 seqlen=64,
                 kind='xyz',
                 stride=1
    ):
        super().__init__()
        
        self.datapath = datapath
        self.seqlen = seqlen
        self.max_len = seqlen

        self.stride = stride
        
        self.n_joints = 8
        
        self._load_data()

    def _load_data(self):        
        data = np.load(self.datapath, allow_pickle=True)[()]
        joints = [v['keypoints'].transpose((1, 0, 3, 2)) for v in data['annotator-id_0'].values()]
        #([n_frames, 2-mice, 2, 7)]

        joints_aligned = []
        self.n_frames = 0
        for i in tqdm(range(len(joints))):
            for j in range(2):
                aligned = align_pose2d(joints[i][j])[0] / 2.0
                maxlen = self.seqlen * (aligned.shape[0] // self.seqlen)
                joints_aligned.append(aligned[:maxlen])
        
        self.joints = torch.from_numpy(np.concatenate(joints_aligned, 0)).float()
        print(f"Dataset: {self.joints.shape}")
    
    def __len__(self):
        return self.joints.shape[0] // self.seqlen
    
    def __getitem__(self, index):
        motion = self.joints[index*self.stride:index*self.stride+self.seqlen]
        motion = motion.reshape((motion.shape[0], -1))
        
        return motion, None
    

if __name__ == "__main__":
    dataset = CalMS21()
    print(len(dataset), dataset[0][0].shape)