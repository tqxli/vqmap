from torch.utils.data import Dataset
import torch
import numpy as np
import os


class MocapTokenDataset(Dataset):
    def __init__(self, datapath, seqlen,
                 subjects, groups=[0, 1, 2],
                 train=True,
                 num_train_samples=10000, 
                 social=False):
        super().__init__()

        self.datapath = datapath
        self.seqlen = seqlen
        self.seqlen_min = seqlen // 4
        assert os.path.isfile(datapath), "the specified data file does not exist"

        self.subjects = subjects
        self.groups = groups
        self.social = social
        self._load_data()
        
        self.train = train
        self.num_train_samples = num_train_samples
        self.val_sampling = seqlen
        
    def _load_data(self):
        tokens = np.load(self.datapath, allow_pickle=True)[()]
        if 'code_indices' in tokens:
            tokens = tokens["code_indices"]
            truncation_len = int(np.floor(tokens.shape[0] / 30)) * 30
            tokens = tokens[:truncation_len]
        else:
            assert 'top' in tokens
            assert 'bottom' in tokens
            tokens_t = tokens['top']
            tokens_b = tokens['bottom']
            truncation_len = int(np.floor(tokens_t.shape[0] / 30)) * 30
            tokens_t = tokens_t[:truncation_len]
            tokens_b = tokens_b[:truncation_len]
            tokens_t = tokens_t[None].repeat(tokens_b.shape[-1], 1)
            tokens_t = tokens_t.reshape((-1))
            tokens_b = tokens_b.reshape((-1))
            tokens = tokens_b * 16 + tokens_t
        print(tokens.shape)
            # tokens = tokens_t * 8 + tokens_b

        tokens = tokens.reshape((5, 6, -1,))
        n_tokens_per_exp = tokens.shape[-1]
        tokens = tokens[self.groups][:, self.subjects]
        tokens = tokens.reshape((-1, n_tokens_per_exp))
        print(f"Token Dataset: {tokens.shape}")

        self.tokens = tokens
        self.num_exps = tokens.shape[0]

        if self.social:
            self.tokens = torch.stack([self.tokens[0::2], self.tokens[1::2]], dim=1)

        self.max_chunks = self.tokens.shape[-1] // self.seqlen
        self.max_token_per_exp = self.tokens.shape[-1] - self.seqlen

    def __len__(self):
        if self.train:
            return self.num_train_samples
        return self.num_exps * (self.tokens.shape[-1] // self.val_sampling)

    def __getitem__(self, idx):
        if self.train:
            # random sample from training data
            expid = np.random.choice(self.num_exps)
            tokenid = np.random.choice(self.max_token_per_exp)
        else:
            # for validation, fix samples to be evaluated
            expid = idx // (self.tokens.shape[-1] // self.val_sampling)
            tokenid = idx % (self.tokens.shape[-1] // self.val_sampling)

        rand_seqlen = self.seqlen #np.random.randint(self.seqlen_min, self.seqlen)
        # if self.social:
        #     tokens = self.tokens[expid, :, tokenid:(tokenid+self.seqlen)]#[start:start+rand_seqlen]

        tokens = self.tokens[expid, tokenid:(tokenid+rand_seqlen)]

        # Fill in missing timesteps (if any)
        t_missing = self.seqlen - tokens.shape[-1]
        if t_missing > 0:
            tokens = torch.cat([tokens, tokens[-1:].repeat(t_missing)])

        return tokens, rand_seqlen


class MocapTokenGTTargets(MocapTokenDataset):
    def __init__(self, datapath, gt_path, seqlen, subjects, **kwargs):
        super().__init__(datapath, seqlen, subjects, **kwargs)
        
        self.gt_path = gt_path
        
        self.tokens = self.tokens[:, ::16]
        self.max_token_per_exp = self.tokens.shape[-1] - self.seqlen
        self._load_gt()
    
    def _load_gt(self):
        tokens = np.load(self.gt_path, allow_pickle=True)[()]
        tokens = np.stack(list(tokens.values()))
        tokens = tokens.reshape((5, 6, -1,))
        tokens = tokens[self.groups][:, self.subjects]
        tokens = tokens[:, :, :self.tokens.shape[-1]+1]
        tokens = tokens.reshape((-1, tokens.shape[-1]))
        print(f"GT motifs: {tokens.shape}")
        self.gt = tokens
    
    def __getitem__(self, idx):
        if self.train:
            # random sample from training data
            expid = np.random.choice(self.num_exps)
            tokenid = np.random.choice(self.max_token_per_exp)
        else:
            # for validation, fix samples to be evaluated
            expid = idx // (self.tokens.shape[-1] // self.val_sampling)
            tokenid = idx % (self.tokens.shape[-1] // self.val_sampling)

        rand_seqlen = self.seqlen #np.random.randint(self.seqlen_min, self.seqlen)

        tokens = self.tokens[expid, tokenid:(tokenid+rand_seqlen)]
        gt = self.gt[expid, tokenid+rand_seqlen+1]
        # # Fill in missing timesteps (if any)
        # t_missing = self.seqlen - tokens.shape[-1]
        # if t_missing > 0:
        #     tokens = torch.cat([tokens, tokens[-1:].repeat(t_missing)])

        return tokens, gt
 

if __name__ == "__main__":
    dataset = MocapTokenDataset(
        datapath='/home/tianqingli/dl-projects/duke-cluster/tqxli/crossmodal/experiments/ablation_studies/ratlone/vq_nb64_d32_t16_commit0.02/SCN2A_WK1_n25308_results.npy',
        seqlen=256,
        subjects=[0, 1, 2, 3, 4],
        )
    print(len(dataset))
    sample = dataset[100]
    print(sample[0].shape, sample[1])
    
    dataset = MocapTokenDataset(
        datapath='/home/tianqingli/dl-projects/duke-cluster/tqxli/crossmodal/experiments/ablation_studies/ratlone/vq_nb64_d32_t16_commit0.02/SCN2A_WK1_n25308_results.npy',
        seqlen=256,
        subjects=[5],
        train=False
    )
    print(len(dataset))
    sample = dataset[0]
    print(sample[0].shape, sample[1])