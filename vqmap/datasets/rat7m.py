import torch
from torch.utils.data import Dataset
import numpy as np
import random

# SpineM(0), SpineF(1), HeadF(2), HeadB(3), HeadL(4),
# SpineL(5),
# HipL(6), HipR(7), ElbowL(8), ArmL(9), ShoulderL(10),
# ShoulderR(11), ELbowR(12), ArmR(13),
# KneeR(14), KneeL(15), ShinL(16), ShinR(17)
REORDER_INDICES = [
    0, 1, 4, 3, 2, 5,
    10, 8, 9,
    11, 12, 13,
    6, 15, 16,
    7, 14, 17,
]
# spineM, spineF, headL, headB, headF, spineL, --> 6
# shoulderL, elbowL, armL, --> 3
# shoulderR, elbowR, armR, --> 3
# hipL, kneeL, shinL, --> 3
# hipR, kneeR, shinR --> 3
# --> total = 18
kinematic_tree = [
    [0, 1, 4, 3, 2],
    [0, 5],
    [1, 6, 7, 8], [1, 9, 10, 11],
    [5, 12, 13, 14], [5, 15, 16, 17]
]

class Rat7M(Dataset):
    def __init__(
        self,
        datapath='/media/mynewdrive/datasets/rat7m/annotations/chunk_preproc_full.npy',
        seqlen=64,
        stride=1,
        kind='xyz',
    ):
        super().__init__()
        
        self.datapath = datapath
        self.seqlen = seqlen
        self.max_len = seqlen
        self.num_keypoints = 20 - 2
        self.kind = kind
        
        self.stride = stride
        
        self.sampling = 'conseq'
        self.sampling_step = 1
        
        self._load_data()
        
    # def _load_data(self):
    #     data = np.load(self.datapath, allow_pickle=True)[()]
    #     pose3d = data["table"]["3D_keypoints"]
    #     day_idx, subject_idx = data["table"]["day_idx"], data["table"]["subject_idx"]
    #     days, subjects = np.unique(day_idx), np.unique(subject_idx)
        
    #     self.data = {}
    #     self.num_samples = []
    #     for d, s in zip(days, subjects):
    #         exp = f"{d}-{s}"
    #         mapping_indices = np.where(
    #             (day_idx == d) & (subject_idx == s)
    #         )[0]
    #         self.data[exp] = pose3d[mapping_indices]
    #         self.num_samples.append(len(mapping_indices) - self.seqlen + 1)

    #     print("Dataset: {}".format(pose3d.shape))
        
    def _load_data(self):
        data = np.load(self.datapath, allow_pickle=True)[()]
        self.pose3d, self.num_samples = [], []
        # for v in data.values():
        #     self.pose3d += v
        #     self.num_samples.append(len(v))
        if self.kind == 'xyz':
            datakind = 'joints3D'
            self.pose3d = [pose/10 for pose in data[datakind]]
        elif self.kind == 'quat':
            datakind = 'quaternion'
            self.pose3d = [pose for pose in data[datakind]]
        elif self.kind == 'cont6d':
            datakind = 'cont6d'
            self.pose3d = [pose for pose in data[datakind]]

        # drop offsets keypoints
        mask = list(np.arange(0, 6)) + list(np.arange(8, 20))
        self.pose3d = [pose[:, mask][:, REORDER_INDICES] for pose in self.pose3d]
        print("RAT7M Dataset: {} shape {}".format(len(self.pose3d), self.pose3d[0].shape))
           
        self.pose3d = [np.reshape(pose, (pose.shape[0], -1)) for pose in self.pose3d]
        self._num_frames = [poseseq.shape[0] for poseseq in self.pose3d]
    
    def __len__(self):
        return len(self.pose3d)
    
    # def __getitem__(self, index):
    #     cumsum, expid = 0, 0
    #     for num in self.num_samples:
    #         if index <= cumsum + num:
    #             break
    #         cumsum += num
    #         expid += 1
        
    #     exp = list(self.data.keys())[expid]
    #     offset = index - cumsum

    #     motion = self.data[exp][offset:offset+self.seqlen]
    #     motion = motion.reshape((motion.shape[0], -1))
        
    #     return motion, 0
    
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

        inp = self.get_pose_data(data_index, frame_ix)
        return inp
    
    def get_pose_data(self, data_index, frame_ix):
        poseseq = self.pose3d[data_index][frame_ix]
        
        return poseseq
    
    def __getitem__(self, index):
        return self._get_item_data_index(index), 0
        
    

if __name__ == "__main__":
    dataset = Rat7M()
    sample = dataset[0]
    print(sample[0].shape)
    
    from vqmap.utils.visualize import visualize
    
    # anim = visualize(
    #     [sample[0].unsqueeze(0).numpy()], 64, 'test_rat7m.mp4', ['GT']
    # )