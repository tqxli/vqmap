from vqmap.datasets.base import *


class MotionTokenDataset(MocapContBase):
    def __init__(self, datapath, cfg):
        super().__init__(datapath, cfg)
        
        self.train = cfg.train
        self.val_sampling = cfg.get("val_sampling", 10)
        self.num_train_samples = cfg.get("num_train_samples", 20000)
    
    def _load_data(self):
        data = np.load(self.datapath, allow_pickle=True)[()]
        datapaths = data["datapath"]
        self.num_exps = len(datapaths)
        tokens = data["code"]
        tokens = tokens[::self.downsample] * self.scale

        self.tokens = torch.from_numpy(tokens).reshape(self.num_exps, -1)
        logger.info(f"Dataset: {self.tokens.shape}")
        self.max_token_per_exp = self.tokens.shape[-1] - self.seqlen
    
    def _load(self, datapath):
        data = np.load(datapath, allow_pickle=True)[()]
        return data["code"]
    
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

        tokens = self.tokens[expid, tokenid:(tokenid+self.seqlen)]

        # Fill in missing timesteps (if any)
        t_missing = self.seqlen - tokens.shape[-1]
        if t_missing > 0:
            tokens = torch.cat([tokens, tokens[-1:].repeat(t_missing)])

        return tokens, self.seqlen


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load('configs/dataset_token.yaml')
    cfg.dataset.train = True
    datapath = cfg.dataset.root
    dataset = MotionTokenDataset(datapath, cfg.dataset)
    poseseq, _ = dataset[0]
    print(poseseq.shape)
    print(f"Total samples: {len(dataset)}")