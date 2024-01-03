from torch.utils.data import DataLoader, Subset
import os
from loguru import logger
from vqmap.datasets.base import *
from vqmap.datasets.mocap2d import *
from vqmap.datasets.mocap3d import *

__all__ = ['initialize_dataset']


def retrieve_datapaths(cfg):
    if os.path.isfile(cfg.root):
        return cfg.root, None
    return globals().get(f'_retrieve_dannce_{cfg.split.mocap_type}')(cfg.root)

def _retrieve_dannce_lone(root):
    candidates = sorted([p for p in os.listdir(root) if p[0].isdigit()])
    datapaths = [
        os.path.join(root, dp, 'SDANNCE', 'bsl0.5_FM', 'save_data_AVG0.mat')
        for dp in candidates
    ]
    datapaths = [dp for dp in datapaths if os.path.exists(dp)]
    return datapaths, candidates

def _retrieve_dannce_soc(root):
    candidates = sorted([p for p in os.listdir(root) if p[0].isdigit()])
    datapaths = []
    for dp in candidates:
        rat1_path = os.path.join(root, dp, 'SDANNCE', 'bsl0.5_FM_rat1', 'save_data_AVG0.mat')
        rat2_path = os.path.join(root, dp, 'SDANNCE', 'bsl0.5_FM_rat2', 'save_data_AVG0.mat')
        if not os.path.exists(rat1_path):
            continue
        if not os.path.exists(rat2_path):
            continue
        datapaths += [rat1_path, rat2_path]
    return datapaths, candidates

def _initialize_dataset(cfg, split='train'):
    dataset_name = cfg.name
    dataset_cls = globals().get(dataset_name, None)
    assert dataset_cls, f"The specified dataset class {dataset_name} was not found."
    
    datapaths, candidates = retrieve_datapaths(cfg)
    
    datapaths = datapaths[:12]
    candidates = candidates[:12]
    
    # split dataset into train and valid:
    # if by subjects/groups, split at datapaths
    # if by fraction, split after dataset is initialized (torch subset)
    split_method = cfg.split.method
    assert split_method in ['fraction', 'subject']
    logger.info(f'Split dataset by {split_method}')
    
    if (split_method == 'subject') and (isinstance(datapaths, list)) and (split != 'inference'):
        split_ids = cfg.split.get(f'split_ids_{split}')
        indices = []
        for id in split_ids:
            indices += [i for i, cand in enumerate(candidates) if id in cand]
        indices = sorted(indices)
        datapaths = [datapaths[idx] for idx in indices]

    # initialize dataset
    dataset = dataset_cls(datapaths, cfg)
    
    if (split_method == 'fraction') and (split != 'inference'):
        train_frac = cfg.split.frac
        cutoff = int(train_frac * len(dataset))
        indices = np.random.permutation(len(dataset))
        indices = indices[:cutoff] if split == 'train' else indices[cutoff:]
        
        dataset = Subset(dataset, indices[:cutoff])
    
    # initialize dataloader
    return get_dataloader(
        dataset, cfg.dataloader,
        batch_size=cfg.dataloader.batch_size if split == 'train' else cfg.dataloader.eval_batch_size,
        shuffle=(split == 'train')
    )

def collate_fn_mocap(samples):
    motion, actions = [], []
    for sample in samples:
        motion.append(sample[0])
        actions.append(sample[1])
    motion = np.stack(motion, 0)
    motion = torch.from_numpy(motion).float()
    
    batch_size, seqlen = motion.shape[0], motion.shape[1]
    
    batch = {
        "motion": motion,
        "action": actions,
        "length": [seqlen]*batch_size,
        "ref": motion.clone()
    }
    return batch

def get_dataloader(dataset, cfg, batch_size, shuffle=True):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        collate_fn=collate_fn_mocap
    )
    return dataloader

def initialize_dataset(cfg, splits=['train', 'val']):
    dataloaders = {}
    for split in splits:
        dataloaders[split] = _initialize_dataset(cfg, split=split)
    infostr = ':'.join([f'{split} {dataloaders[split].dataset.__len__()}' for split in splits])
    logger.info(infostr)
    return dataloaders

if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/dataset.yaml')
    dataloaders = initialize_dataset(cfg.dataset)