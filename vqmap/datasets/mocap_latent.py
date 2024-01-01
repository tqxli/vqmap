from torch.utils.data import Dataset
import torch
import numpy as np
import os


class MocapLatentDataset(Dataset):
    def __init__(self, datapath, seqlen, groups):
        super().__init__()

        self.datapath = datapath
        self.seqlen = seqlen
        assert os.path.isfile(datapath), "the specified data file does not exist"
        
        self.groups = groups
        
        self._load_data()

        self.val_sampling = seqlen
        
    def _load_data(self):
        data = np.load(self.datapath, allow_pickle=True)[()]
        inputs, tokens = data["inputs"], data["code_indices"]

        n_tokens_per_exp = tokens.shape[0] // 30
        tokens = tokens[:n_tokens_per_exp * 30]
        tokens = tokens.reshape((5, 6, -1, *tokens.shape[1:]))
        tokens = tokens[self.groups]
        tokens = tokens.reshape((-1, *tokens.shape[3:]))
        
        inputs = inputs[:n_tokens_per_exp * 30]
        inputs = inputs.reshape((5, 6, -1, *inputs.shape[1:]))
        inputs = inputs[self.groups]
        inputs = inputs.reshape((-1, *inputs.shape[3:]))
        # inputs = inputs.reshape((*inputs.shape[:2], -1, 3))
        
        print(f"Dataset: {tokens.shape} {inputs.shape}")

        self.latents = torch.from_numpy(tokens)
        self.inputs = inputs

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.inputs[idx], self.latents[idx]
    

if __name__ == "__main__":
    dataset = MocapLatentDataset(
        datapath='experiments/vqvae2_rebuttal/cont_beta1e-4_d64/SCN2A_WK1_n84328_results.npy',
        seqlen=64,
        groups = [0, 1, 2, 3]
        )
    print(len(dataset))
    sample = dataset[100]
    for k, v in sample.items():
        print(k, v.shape)