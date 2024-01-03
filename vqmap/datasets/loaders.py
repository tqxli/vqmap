import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from vqmap.datasets.mocap_ephys import MocapEphys
from vqmap.datasets.mocap import Mocap, MocapUnannotated
from vqmap.datasets.mocap_token import MocapTokenDataset, MocapTokenGTTargets
from vqmap.datasets.rat7m import Rat7M
# from vqmap.datasets.calms import CalMS21
# from vqmap.datasets.moseq2d import Moseq2D
from vqmap.datasets.human import *
from vqmap.datasets.mocap_latent import MocapLatentDataset

mapping = {
    "mocap": Mocap,
    # "rat7m": Rat7M,
    # "calms21": CalMS21,
    # "moseq2d": Moseq2D,
    "humanact12": HumanAct12,
    "uestc": UESTC,
    "oms": OMS
}

def collate_fn(samples):
    neural, motion = [], []
    for sample in samples:
        neural.append(sample[0])
        motion.append(sample[1])
    neural, motion = torch.stack(neural, 0), torch.stack(motion, 0)
    
    seqlen = neural.shape[1]
    batch_size = neural.shape[0]
        
    batch = {
        "neural": neural,
        "motion": motion,
        "length": [seqlen]*batch_size,
        "ref": motion.clone()
    }
    return batch

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

def collate_fn_mocap_noisy(samples):
    motion, actions = [], []
    for sample in samples:
        motion.append(sample[0])
        actions.append(sample[1])
    motion = np.stack(motion, 0)
    motion = torch.from_numpy(motion).float()
    
    batch_size, seqlen = motion.shape[0], motion.shape[1]

    noise = torch.randn_like(motion) * 4
    noisy_motion = motion.clone() + noise
    
    batch = {
        "motion": noisy_motion,
        "action": actions,
        "length": [seqlen]*batch_size,
        "ref": motion.clone()
    }
    return batch
    

def _get_loader(
    name,
    config,
    root, sessions, seqlen, split,
    batch_size, train, num_workers,
    logger,
    sampling=True, num_valid=1000,
    train_frac=0.9, inference=False,
    datakind='xyz'
):
    if name == "mocapephys":
        if inference:
            dataset = MocapEphys(root, sessions, seqlen, split, frac=(0.0, 1.0))
        else:
            frac = (0.2, 1) if train else (0, 0.2)
            dataset = MocapEphys(root, sessions, seqlen, split, frac=frac)

        if sampling:
            # will use custom data sampler during training (entire dataset too large)
            dataloader = dataset
        else:
            # np.random.seed(1024)
            # indices = np.arange(num_valid)
            # dataset = Subset(dataset, indices)        
            shuffle = True if train and (not inference) else False
            print("Do shuffle: {}".format(shuffle))
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
    elif name in list(mapping.keys()):
        dataset_cls = mapping[name]
        dataset = dataset_cls(root, seqlen, datakind)
        np.random.seed(1024)

        if inference:
            print("Running inference over the complete dataset")
            cutoff = len(dataset)
            indices = np.arange(len(dataset))
        else:
            # further divide into train and validation during training
            train_frac = 0.9
            cutoff = int(train_frac * len(dataset))
            indices = np.random.permutation(len(dataset))

        if train:
            dataset = Subset(dataset, indices[:cutoff])
        else:
            dataset = Subset(dataset, indices[cutoff:])
        print(f"Dataset size: {len(dataset)}")
        
        shuffle = True if train and (not inference) else False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_mocap,
            pin_memory=True,
            drop_last=True
        )
    elif name == "mocap_unannot":
        def retrieve_datapaths(root, is_train):
            datapath = []
            social_data = ('SOC' in root)

            if social_data:
                candidates = sorted([p for p in os.listdir(root) if p.startswith('2022')])
                if not is_train:
                    candidates = candidates[6:7]
                elif not inference:
                    candidates = candidates[:3]
                
                for dp in candidates:
                    rat1_path = os.path.join(root, dp, 'SDANNCE', 'bsl0.5_FM_rat1', 'save_data_AVG0.mat')
                    rat2_path = os.path.join(root, dp, 'SDANNCE', 'bsl0.5_FM_rat2', 'save_data_AVG0.mat')
                    if not os.path.exists(rat1_path):
                        continue
                    if not os.path.exists(rat2_path):
                        continue
                    datapath += [rat1_path, rat2_path]
            else:
                candidates = sorted([p for p in os.listdir(root) if p.startswith('2022')])
                assert len(candidates) == 30
                splits = {
                    "subjects_train": config.get("subjects_train", [0, 1, 2, 3, 4, 5]),
                    "subjects_valid": config.get("subjects_valid", [0, 1, 2, 3, 4, 5]),
                    "groups_train": config.get("groups_train", [0, 1, 2, 3]),
                    "groups_valid": config.get("groups_valid", [4]),
                }
                
                # save in config for future reference
                for k, v in splits.items():
                    config[k] = v
                
                subjects = splits["subjects_train"] if train else splits["subjects_valid"]
                groups = splits["groups_train"] if train else splits["groups_valid"]
                indices = []
                for g in groups:
                    for s in subjects:
                        indices.append(6*g+s)
                print("Dataset group: ", indices)
                if not inference:
                    candidates = [candidates[idx] for idx in indices]
                
                # two rounds of experiments M1-M6 x2
                # candidates = sorted([p for p in os.listdir(root) if p.startswith('2022')])#[:18]
                # subjects = ['M1', 'M2', 'M3', 'M4', 'M5'] if is_train else ['M6']
                # if inference:
                #     subjects = [f'M{i+1}' for i in range(6)]
                # candidates = [candidate for candidate in candidates if candidate.split('_')[-1] in subjects]
                # if is_train:
                #     candidates = candidates[:18]
                # else:
                #     candidates = candidates[-6:]
                
                for dp in candidates:
                    datapath.append(os.path.join(root, dp, 'SDANNCE', 'bsl0.5_FM', 'save_data_AVG0.mat'))
            
            return datapath
        split_name = "train" if train else "validation"
        logger.log(f"Prepare {split_name} ...")
        datapath = retrieve_datapaths(root, train)
        datapath = [dp for dp in datapath if os.path.exists(dp)]
        logger.log("Recording number: {}".format(len(datapath)))
        
        # CHANGES: stride becomes half of seqlen!
        stride = seqlen // 2 if not inference else 1 #seqlen
        # stride = seqlen // 2
        # stride = 1 #seqlen
        config["stride"] = stride
        logger.log("Stride: ", stride)
        
        dataset = MocapUnannotated(datapath, seqlen, stride=stride, kind=datakind, keep_keypoints=config.get("keep_keypoints", None))
        logger.log(f"Dataset size: {len(dataset)}")

        shuffle = True if train and (not inference) else False
        logger.log("Do shuffle: ", shuffle)

        noisy_dataloader = config.get("noisy_dataloader", False)
        collate_fn_cls = collate_fn_mocap_noisy if noisy_dataloader else collate_fn_mocap
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_cls,
            pin_memory=True,
            drop_last=(not inference),
        )
    elif name == "mocap_mouse22":
        datapath = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.mat')])
        print("Valid recording number: {}".format(len(datapath)))
        
        split = [1] if not train else [0, 2, 3, 4]
        if inference:
            split = [0, 1, 2, 3, 4]
        datapath = [datapath[idx] for idx in split]
        
        stride = seqlen
        dataset = MocapUnannotated(datapath, seqlen, stride=stride, kind=datakind, downsample=2, scale=2)
        print(f"Dataset size: {len(dataset)}")
        shuffle = True if train and (not inference) else False
        print("Do shuffle: ", shuffle)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_mocap,
            pin_memory=True)
    elif name == "mocap_token":
        splits = {
            "subjects_train": config.get("subjects_train", [0, 1, 2, 3, 4,]),
            "subjects_valid": config.get("subjects_valid", [5]),
            "groups_train": config.get("groups_train", [0, 1, 2]),
            "groups_valid": config.get("groups_valid", [0, 1, 2]),
        }
        
        subjects = splits["subjects_train"] if train else splits["subjects_valid"]
        groups = splits["groups_train"] if train else splits["groups_valid"]
        dataset = MocapTokenDataset(
            root, seqlen, 
            subjects, groups,
            train=train,
            num_train_samples=config.get("num_train_samples", 10000),
        )
        shuffle = True if train and (not inference) else False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    elif name == "mocap_latent":
        dataset = MocapLatentDataset(
            root, seqlen, 
            groups=[0, 1, 2, 3] if train else [4],
            subjects=[0, 1, 2, 3, 4, 5]
        )
        shuffle = True if train and (not inference) else False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn_mocap
        )
    elif name == "mocap_token_target":
        dataset = MocapTokenGTTargets(
            root, config.get("gt_path"), seqlen, 
            groups=[0, 1, 2, 3] if train else [4],
            subjects=[0, 1, 2, 3, 4, 5]
        )
        shuffle = True if train and (not inference) else False
        def _collate_fn(samples):
            motion, actions = [], []
            for sample in samples:
                motion.append(sample[0])
                actions.append(sample[1])
            motion = np.stack(motion, 0)
            motion = torch.from_numpy(motion).float().unsqueeze(-1)

            actions = torch.tensor(actions)
            
            batch_size, seqlen = motion.shape[0], motion.shape[1]
            
            batch = {
                "motion": motion,
                "y": actions,
                "length": [seqlen]*batch_size,
                "ref": motion.clone()
            }
            return batch
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_collate_fn
        )        
        
    return dataloader


def prepare_dataloaders(dataloader_config, logger, num_workers=1, split=None):
    dataset_name = dataloader_config["name"]
    
    root = dataloader_config["root"]
    seqlen = dataloader_config["seqlen"]
    batch_size = dataloader_config['batch_size']
    eval_batch_size = dataloader_config.get('eval_batch_size', batch_size)
    datakind = dataloader_config.get('kind', 'xyz')
    
    do_inference = dataloader_config.get("inference", False)

    dataloaders = {}
    
    if split is None:
        dataset_list = ['train', 'val']
    else:
        dataset_list = [split]

    if 'train' in dataset_list:
        dataloaders['train'] = _get_loader(
            dataset_name,
            dataloader_config,
            root, dataloader_config.get("train_sessions", None),
            seqlen, split='lone', batch_size=batch_size, train=True,
            num_workers=num_workers,
            sampling=dataloader_config.get("sampling", True),
            inference=do_inference,
            datakind=datakind,
            logger=logger,
        )

    if 'val' in dataset_list:
        dataloaders['val'] = _get_loader(
            dataset_name,
            dataloader_config,
            root, dataloader_config.get("valid_sessions", None),
            seqlen, split='lone', batch_size=eval_batch_size, train=False,
            num_workers=num_workers,
            sampling=False,
            num_valid=dataloader_config.get("num_valid", 1000),
            inference=do_inference,
            datakind=datakind,
            logger=logger,
        )

    return dataloaders