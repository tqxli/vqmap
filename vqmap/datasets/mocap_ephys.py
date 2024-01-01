import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
from vqmap.datasets.utils import align_pose

_SESSIONS = [26, 27, 28, 29, 30, 31]
FPS = 50
_CUTOFFS = [80, 80, 80, 80, 80, 80] #[110, 98, 93, 84, 93, 91]
_CUTOFFS = [c*FPS*60 for c in _CUTOFFS]

class MocapEphys(Dataset):
    """Animal motion capture dataset with simultaneous electrophysiology recordings
    """
    def __init__(
        self, root, sessions,
        seqlen,
        split='lone',
        bin=1,
        frac=(0, 1),
        drop_neurons=False, drop_threshold=1e3,
    ):
        super().__init__()
        
        self.root = root
        for sess in sessions:
            assert sess in _SESSIONS
        self.sessions = sessions
        self.split = split
        self.seqlen = seqlen
        self.bin = bin
        self.frac = frac
        
        self.drop_neurons = drop_neurons
        self.drop_threshold = drop_threshold
        
        self._setup_neural()
        self._load_sessions()

    def _setup_neural(self):
        self.bins_before = bins_before = 8
        self.bins_current = bins_current = 16 
        self.bins_after = bins_after = 8
        self.surrounding_bins = bins_before + bins_current + bins_after
    
    def _process(self, spike_counts, pose3d):
        num_samples, num_neurons = spike_counts.shape[:2]
        num_samples_valid = num_samples-self.bins_before-self.bins_after
        X = np.empty((num_samples_valid, self.surrounding_bins, num_neurons))
        X[:] = np.NaN

        for start_idx in range(num_samples_valid):
            end_idx = start_idx + self.surrounding_bins
            X[start_idx] = spike_counts[start_idx:end_idx,:]
        
        return X
 
    def _load_sessions(self):
        spikes, poses = [], []
        print(f"Loading sessions {self.sessions}")
        for id, session in enumerate(self.sessions):
            session_path = os.path.join(self.root, f'session{session}_ephys_full.npy')
            data = np.load(session_path, allow_pickle=True)[()]
            spike_counts = data["spike_counts"]
            pose3d = data["pose3d"][:, 0]
            
            if self.split == "lone":
                cutoff = _CUTOFFS[id]
                spike_counts, pose3d = spike_counts[:cutoff], pose3d[:cutoff]

            pose3d = self._align_data(pose3d)
            
            spikes.append(spike_counts)
            poses.append(pose3d)
        
        n_neurons = [data.shape[-1] for data in spikes]
        n_neuron_min = min(n_neurons)
        spikes = [spike[:, :n_neuron_min] for spike in spikes]
        neural = torch.from_numpy(np.concatenate(spikes, axis=0)).float()
        
        if self.drop_neurons:
            n_spikes_per_neuron = [neural[:, i].sum() for i in range(neural.shape[-1])]
            masking = np.array(n_spikes_per_neuron) > self.drop_threshold
            neural = neural[:, masking]
        
        self.neural = neural
        
        motion = np.concatenate(poses, axis=0)
        self.motion = torch.from_numpy(motion).float()

        assert self.neural.shape[0] == self.motion.shape[0]
        
        start = self.motion.shape[0] * self.frac[0]
        end = self.motion.shape[0] * self.frac[-1]
        start, end = int(start), int(end)
        self.motion = self.motion[start:end]
        self.neural = self.neural[start:end]
        
        self.num_neurons = self.neural.shape[-1]
        self.motion_dim = self.motion.shape[-1] * self.motion.shape[-2]
        
        # num_samples = self.neural.shape[0] - self.bins_before - self.bins_after
        self.n_frames = self.neural.shape[0]
        
        print("Sample number: {}".format(self.n_frames))
        self.n_chunks = self.n_frames // self.seqlen
        
        # further bin the spikes
        # self.neural = self.neural.reshape(-1, self.bin, self.neural_dim).sum(1)
        print("After binning: ", self.neural.shape)
    
    def normalize_motion(self, seq):
        # print("Normalize poses ...")
        return align_pose(seq)

    def _align_data(self, joints3D):
        aligned = []
        for seq in tqdm(joints3D.reshape(-1, self.seqlen, *joints3D.shape[1:])):
            aligned.append(align_pose(torch.from_numpy(seq).float())[0].numpy())
            
        aligned = np.concatenate(aligned, axis=0)
        return aligned

    def sample(self):
        fr_start = np.random.randint(self.n_frames - self.seqlen)
        fr_end = fr_start + self.seqlen
        
        motion = self.motion[fr_start:fr_end]
        motion = motion.flatten(1)
        
        return self.neural[fr_start//self.bin:fr_end//self.bin], motion

    def _batch(self, neural, motion):
        return {
            "neural": neural,
            "motion": motion,
            "length": [self.seqlen]*neural.shape[0],
            "ref": motion
        }
    
    def sampling_generator(self, num_samples=1000, batch_size=16):
        for i in range(num_samples // batch_size):
            sample_n, sample_m = [], []
            for _ in range(batch_size):
                sample_i = self.sample()
                sample_n.append(sample_i[0])
                sample_m.append(sample_i[1])
            sample_n, sample_m = torch.stack(sample_n, dim=0), torch.stack(sample_m, dim=0)
            yield self._batch(sample_n, sample_m)
    
    def __len__(self):
        return self.n_frames-self.seqlen+1 #self.n_chunks
    
    def __getitem__(self, index):
        # sample_range = np.arange(index*self.seqlen, (index+1)*self.seqlen)
        sample_range = np.arange(self.seqlen)+index
        neural = self.neural[sample_range]
        motion = self.motion[sample_range]
        motion = motion.flatten(1)
        
        return neural, motion
    

# class EphysMotionTokens(Dataset):
    

if __name__ == "__main__":
    root = '/media/mynewdrive/datasets/dannce_ephys'
    seqlen = 64
    sessions = [26, 27]
    binsize = 1
    dataset = MocapEphys(root, sessions, seqlen, bin=binsize)
    print(len(dataset))
    neural, motion = dataset[3000]
    print(neural.shape, motion.shape)
    
    # import matplotlib.pyplot as plt
    # plt.imshow(neural)
    # plt.show()
