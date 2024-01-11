import torch
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from copy import deepcopy
from loguru import logger
from omegaconf import OmegaConf
from vqmap.datasets.base import MocapChunkBase, MocapContBase
from vqmap.datasets.mocap2d import MocapCont2D, MoSeqDLC2D
from vqmap.datasets.mocap3d import *
from vqmap.datasets.motion_token import MotionTokenDataset


__all__ = ['initialize_dataset']


def retrieve_datapaths(cfg):
    if os.path.isfile(cfg.root):
        return cfg.root, []
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


def _retrieve_dannce_token(root):
    candidates = sorted([p for p in os.listdir(root) if p[0].isdigit()])
    datapaths = [
        os.path.join(root, dp, 'vq_inference.npy')
        for dp in candidates
    ]
    datapaths = [dp for dp in datapaths if os.path.exists(dp)]
    return datapaths, candidates


def _initialize_dataset(cfg, split='train'):
    cfg_dataset = cfg.dataset
    dataset_name = cfg_dataset.name
    cfg_dataset.train = (split == 'train')
    dataset_cls = globals().get(dataset_name, None)
    assert dataset_cls, f"The specified dataset class {dataset_name} was not found."
    
    datapaths, candidates = retrieve_datapaths(cfg_dataset)
    
    if isinstance(datapaths, list):
        datapaths = datapaths[:12]
        candidates = candidates[:12]
    
    # split dataset into train and valid:
    # if by subjects/groups, split at datapaths
    # if by fraction, split after dataset is initialized (torch subset)
    split_method = cfg_dataset.split.method
    
    if (split_method == 'subject') and (isinstance(datapaths, list)) and (split != 'inference'):
        split_ids = cfg_dataset.split.get(f'split_ids_{split}')
        indices = []
        for id in split_ids:
            indices += [i for i, cand in enumerate(candidates) if id in cand]
        indices = sorted(indices)
        datapaths = [datapaths[idx] for idx in indices]

    # initialize dataset
    dataset = dataset_cls(datapaths, cfg_dataset)
    
    if (split_method == 'fraction') and (split != 'inference'):
        train_frac = cfg_dataset.split.frac
        cutoff = int(train_frac * len(dataset))
        np.random.seed(cfg.seed)
        indices = np.random.permutation(len(dataset))
        indices = {
            'train': indices[:cutoff],
            'val':  indices[cutoff:]
        }
        dataset = {k:Subset(dataset, ind) for k, ind in indices.items()}
    else:
        dataset = {split:dataset}

    do_shuffle = (split == 'train')
    logger.info(f'Shuffle data: {do_shuffle}')

    # initialize dataloader
    return get_dataloader(
        dataset, cfg.dataloader,
        batch_size=cfg.dataloader.batch_size if split == 'train' else cfg.dataloader.eval_batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate_fn_mocap if 'token' not in cfg_dataset.split.mocap_type else None
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


def get_dataloader(datasets, cfg, batch_size, shuffle=True, collate_fn=None, drop_last=False):
    dataloaders = {}
    for split, dataset in datasets.items():
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.get('pin_memory', True),
            collate_fn=collate_fn,
            drop_last=drop_last
        )

    return dataloaders


def initialize_dataset_single(cfg, splits):    
    dataloaders = {}
    for split in splits:
        dataloaders_split = _initialize_dataset(cfg, split=split)
        dataloaders.update(dataloaders_split)
    infostr = ':'.join([f'{split} {dataloaders[split].dataset.__len__()}' for split in dataloaders])
    logger.info(f"Dataset splits: {infostr}")
    return dataloaders


def initialize_dataset_multi(cfg, splits):
    dataloaders = []
    for cfg_dataset in cfg.dataset:
        cfg_new = deepcopy(cfg)
        cfg_new.update(cfg_dataset)
        loaders = initialize_dataset_single(cfg_new, splits)
        dataloaders.append(loaders)
    
    dataloaders = {k: [loaders[k] for loaders in dataloaders] for k in dataloaders[0]}
    return dataloaders


def initialize_dataset(cfg, splits=['train', 'val']):
    split_method = cfg.dataset.split.method
    assert split_method in ['fraction', 'subject']
    logger.info(f'Split dataset by {split_method}')
    
    if (split_method == 'fraction') and (splits != ['inference']):
        splits = ['train']
    
    if OmegaConf.is_list(cfg.dataset):
        return initialize_dataset_multi(cfg, splits)
    return initialize_dataset_single(cfg, splits)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/dataset.yaml')
    dataloaders = initialize_dataset(cfg.dataset)