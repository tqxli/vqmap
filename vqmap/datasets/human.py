import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import scipy.io as sio
from vqmap.utils.quaternion import *
from vqmap.utils.skeleton import skeleton_initialize
from tqdm import tqdm

humanact12_coarse_action_enumerator = {
    0: "warm_up",
    1: "walk",
    2: "run",
    3: "jump",
    4: "drink",
    5: "lift_dumbbell",
    6: "sit",
    7: "eat",
    8: "turn steering wheel",
    9: "phone",
    10: "boxing",
    11: "throw",
}

class HumanAct12(Dataset):
    def __init__(
        self,
        datapath='/home/tianqingli/dl-projects/ACTOR/data/HumanAct12Poses/humanact12poses.pkl',
        seqlen=64,
        kind='xyz'
    ):
        super().__init__()

        self.datapath = datapath
        self.seqlen = seqlen
        self.max_len = seqlen
        
        self.sampling = 'conseq'
        self.sampling_step = 1
        
        self.kind = kind
        
        self._action_classes = humanact12_coarse_action_enumerator
        self._load_data()
        
    
    def _load_data(self):
        data = np.load(self.datapath, allow_pickle=True)

        pose3d = data['joints3D']
        self.num_joints = pose3d[0].shape[1]
        self.pose3d, self._num_frames = [], []
        for poseseq in pose3d:
            poseseq -= poseseq[:, :1]
            poseseq = np.stack([-poseseq[:, :, 1], poseseq[:, :, 0], poseseq[:, :, 2]], axis=-1)
            
            poseseq = poseseq.reshape((poseseq.shape[0], -1)) * 8
            self.pose3d.append(poseseq)
            self._num_frames.append(poseseq.shape[0])

        self.actions = data['y']
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


class UESTC(HumanAct12):
    def __init__(self,
                 datapath, seqlen, kind='xyz'):
        super().__init__(datapath, seqlen, kind)
        
    def _load_data(self):
        self.joints_idx = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]
        data = np.load(self.datapath, allow_pickle=True)[()]

        pose3d = data['joints3d']
        self.num_joints = pose3d[0].shape[1]
        self.pose3d, self._num_frames = [], []
        for poseseq in pose3d:
            poseseq = np.stack([-poseseq[:, :, 1], poseseq[:, :, 0], poseseq[:, :, 2]], axis=-1)
            poseseq = poseseq[:, self.joints_idx]
            poseseq -= poseseq[:, :1]
            poseseq = poseseq.reshape((poseseq.shape[0], -1)) * 8

            self.pose3d.append(poseseq)
            self._num_frames.append(poseseq.shape[0])
            
        self.actions = np.zeros((len(self.pose3d)))
        assert len(self.pose3d) == len(self.actions)
  

class OMS(HumanAct12):
    def __init__(self,
                 datapath, seqlen, kind='xyz'):
        super().__init__(datapath, seqlen, kind)
        
    def _load_data(self):
        data = np.load(self.datapath, allow_pickle=True)[()]

        self.pose3d = data['joints3D'] * 4
        num_frames, self.num_joints = self.pose3d.shape[:2]
        self._num_frames = num_frames
        self.pose3d = self.pose3d.reshape((num_frames, -1))
    
    def __len__(self):
        return self._num_frames - self.seqlen + 1
    
    def __getitem__(self, index):
        return self.pose3d[index:index+self.seqlen], 0
        


if __name__ == '__main__':
    # dataset = HumanAct12(
    #     datapath='/home/tianqingli/dl-projects/ACTOR/data/HumanAct12Poses/humanact12poses.pkl',
    #     seqlen=64
    # )
    # dataset = UESTC(
    #     datapath='/home/tianqingli/dl-projects/ACTOR/data/uestc/vibe_cache_refined.pkl',
    #     seqlen=64
    # )
    dataset = OMS(
        datapath='/media/mynewdrive/datasets/OMS_Dataset/reformat_asoid/oms_demo.npy',
        seqlen=64
    )
    print(f"Number: {len(dataset)}")
    sample = dataset[0]
    print(sample[0].shape, sample[1])