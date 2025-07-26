from copy import deepcopy
from typing import Dict, List
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, ListConfig
from loguru import logger
from hydra.utils import instantiate

import csbev.dataset.io as io_utils


def filter_by_keys(keys: List[str], candidates: List[str]):
    indices = []
    for key in keys:
        indices += [cid for cid, c in enumerate(candidates) if key in c]
    return indices


def split_datapaths(
    dataroot: str, dataset_type: str, split_cfg: str,
):
    """Scan the dataset root directory, locate the available data files (.mat, .npy etc.), and split them into train and val sets if specified.

    Args:
        dataroot (str): root directory of the dataset, or a compiled data file where the entire dataset can be directly loaded. 
        dataset_type (str): indicator for how the experiment data is organized, e.g., "dannce_lone". Must correspond to a specific data retrieval function defined in `csbev.dataset.io`.
        split_cfg (str): data split configuration. 

    Returns:
        datapaths_split: Dict[split, List[datapath]]
    """
    # multiple data directory paths have been specified； scan the folders and retrieve corresponding datapaths
    if isinstance(dataroot, ListConfig):
        assert hasattr(io_utils, f"_retrieve_{dataset_type}")
        datapaths, candidates = [], []
        for dr in dataroot:
            dp, c = getattr(io_utils, f"_retrieve_{dataset_type}")(dr)
            datapaths += dp
            candidates += c 
    # a compiled data file containing multiple experiments
    elif os.path.isfile(dataroot):
        _data = np.load(dataroot, allow_pickle=True)
        _data = _data[()] if dataroot.endswith('.npy') else _data
        if isinstance(_data, dict) and split_cfg.method == "fraction":
            datapaths = candidates = list(_data.keys())
            del _data
            return {"full": datapaths}
        elif isinstance(_data, dict) and split_cfg.method == "subject":
            datapaths = candidates = list(_data.keys())
        else:
            return {"full": dataroot}
        
        del _data 
    elif os.path.isdir(dataroot):
        assert hasattr(io_utils, f"_retrieve_{dataset_type}")
        datapaths, candidates = getattr(io_utils, f"_retrieve_{dataset_type}")(dataroot)
    else:
        raise ValueError(f"Unknown dataroot type: {dataroot}")

    # assume splitting by subject_ids
    subjects_all = split_cfg.get("split_ids", [])
    train_subjects = split_cfg.get("split_ids_train", [])
    val_subjects = split_cfg.get("split_ids_val", [])

    datapaths_split = {
        "full": [datapaths[i] for i in filter_by_keys(subjects_all, candidates)],
        "train": [datapaths[i] for i in filter_by_keys(train_subjects, candidates)],
        "val": [datapaths[i] for i in filter_by_keys(val_subjects, candidates)],
    }
    datapaths_split = {
        split: paths for split, paths in datapaths_split.items() if len(paths) > 0
    }

    if "full" in datapaths_split:
        # "full" overrides "train" and "val"
        datapaths_split = {"full": datapaths_split["full"]}

    return datapaths_split


def post_hoc_data_split(dataset: Dataset, split_cfg: DictConfig):
    if split_cfg.method == "fraction":
        assert hasattr(
            split_cfg, "train_frac"
        ), "train_frac must be specified if split dataset by fraction"
        train_frac = split_cfg.train_frac
        assert 0 < train_frac < 1, "train_frac must be in (0, 1)"
        cutoff = int(len(dataset) * train_frac)
        indices = np.random.permutation(len(dataset))
        indices_map = {"train": indices[:cutoff], "val": indices[cutoff:]}
        datasets = {
            split: torch.utils.data.Subset(dataset, indices)
            for split, indices in indices_map.items()
        }
    else:
        raise NotImplementedError(f"Split method {split_cfg.method} not implemented")

    return datasets


def prepare_datasets(cfg: DictConfig, splits: List[str] = ["train", "val"]):
    """Loaded datasets specified in the config.
    For each dataset, there exist three possible data loading schemes:
    1. Different data files/paths for train/val. Load them separately and merge.
    2. A single data file containing all data as a Dict[expname, np.ndarray]. Split by keys.
    3. 1/2 + post-hoc data split.

    Args:
        datasets_cfg (DictConfig): Dict[dataset_tag, dataset_cfg].
        splits (list, optional): Defaults to ["train", "val"].

    Returns:
        datasets: Dict[split, Dict[dataset_tag, dataset]]
    """
    if not hasattr(cfg, "datasets"):
        assert hasattr(cfg, "dataset") and hasattr(cfg.dataset, "name")
        datasets_cfg = {cfg.dataset.name: cfg.dataset}
    else:
        datasets_cfg = cfg.datasets
    logger.info(f"Loading {list(datasets_cfg.keys())}")

    datasets = {}
    skeletons = {}
    for tag, _dataset_cfg in datasets_cfg.items():
        dataroot = _dataset_cfg.dataroot
        dataset_type = _dataset_cfg.get("dataset_type", None)

        # Parse paths for train/val/inference
        datapaths = split_datapaths(dataroot, dataset_type, _dataset_cfg.split)
        datapaths = {k: sorted(list(set(v))) for k, v in datapaths.items()}
        if (
            (not os.path.isfile(dataroot))
            and splits == ["full"]
            and "full" not in datapaths
        ):
            datapaths = {"full": sum(list(datapaths.values()), [])}
            datapaths["full"] = sorted(datapaths["full"])
        elif _dataset_cfg.split.method == "fraction":
            datapaths = {"full": datapaths["full"]}
        else:
            datapaths = {k:v for k, v in datapaths.items() if k in splits}

        for split in datapaths.keys():
            dataset_cfg = dict(deepcopy(_dataset_cfg))
            dataset_cfg.pop("split")
            dataset_cfg["dataroot"] = dataroot
            dataset_cfg["datapaths"] = datapaths[split]
            dataset = instantiate(dataset_cfg)

            if split not in datasets:
                datasets[split] = {}
            datasets[split][tag] = dataset

            if tag not in skeletons:
                skeletons[tag] = dataset.skeleton

        if splits == ["full"]:
            logger.info(
                f"Loaded full {tag} for inference only: {len(datasets['full'][tag])} samples"
            )

        elif "full" in datapaths.keys():
            if set(splits) <= set(["train", "val"]):
                _datasets = post_hoc_data_split(
                    datasets["full"][tag], _dataset_cfg.split
                )
                for split, data in _datasets.items():
                    if split not in datasets:
                        datasets[split] = {}
                    
                    datasets[split][tag] = data
                logger.info(
                    f"Loaded {tag}: train:val = {len(datasets['train'][tag])}:{len(datasets['val'][tag])}"
                )
            for k in list(datasets.keys()):
                if k not in splits:
                    datasets.pop(k)

        else:
            split_names = ":".join(datasets.keys())
            num_samples = ":".join([str(len(datasets[split][tag])) for split in datasets.keys()])
            logger.info(
                f"Loaded {tag}: {split_names} = {num_samples} samples"
            )

    # Update cfg with dataset information
    if "train" in datasets:
        out_channels = {}
        for tag, dataset in datasets["train"].items():
            if isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset
            out_channels[tag] = dataset.skeleton.n_keypoints

        cfg.model.decoder["out_channels"] = out_channels

    return datasets, skeletons


def prepare_dataloaders(
    datasets: Dict[str, Dict[str, Dataset]], dataloader_cfg: DictConfig,
):
    """Prepare dataloaders for the datasets. 

    Args:
        datasets (Dict[str, Dict[str, Dataset]]): Dict[split, Dict[dataset_tag, dataset]].
        dataloader_cfg (DictConfig): dataloader configuration. 

    Returns:
        dataloaders: Dict[split, DataLoader]
    """
    dataloaders = {}
    for split, _datasets in datasets.items():
        dataloaders[split] = {}
        for tag, dataset in _datasets.items():
            dataloader = DataLoader(
                dataset, shuffle=(split == "train"), **dataloader_cfg[split],
            )
            dataloaders[split][tag] = dataloader
    return dataloaders


if __name__ == "__main__":
    from hydra import initialize, compose
    from hydra.utils import instantiate

    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="combo.yaml")
        datasets = prepare_datasets(cfg.datasets)
