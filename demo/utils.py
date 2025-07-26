import os
from copy import deepcopy
from typing import Literal
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from hydra import compose, initialize
from hydra.utils import instantiate
import omegaconf
from omegaconf import OmegaConf
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from sklearn.decomposition import PCA
import pandas as pd
import scipy.stats as st

from csbev.dataset.loader import filter_by_keys
from csbev.utils.visualization import reset_rcparams, mm, hide_axes_all, hide_axes_top_right, visualize_pose, make_pose_seq_overlay, make_pose3d_seq_video


BASE_CODEROOT = '/home/tianqingli/dl-projects/vqmap2'


def get_figsize_in_mm(figsize):
    return (figsize[0] * mm, figsize[1] * mm)


def execute_command(command):
    print(" ".join(command))
    os.system(" ".join(command))


def launch_task(exp, additional_args=""):
    base_command = [f"python {BASE_CODEROOT}/csbev/core/inference.py"]
    checkpoint_path = os.path.join(exp, "checkpoints", "model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for experiment {exp}. Skip.")
        return
    
    ckpt = torch.load(checkpoint_path)
    config = ckpt["config"]
    metadata = ckpt["metadata"]
    epoch = metadata["epoch"]
    
    if epoch+1 != config.train.n_epochs:
        print(f"Experiment {exp} incomplete. Skip.")
        return
    
    execute_command(base_command + [f"checkpoint_path={checkpoint_path}"] + additional_args.split(" "))
    
    print("DONE!")


def load_datasets(dataset_loading_params, batch_size=64):
    datasets = {}
    datapaths = {}
    skeletons = {}
    for tag, load_params in dataset_loading_params.items():
        datasets[tag], skeletons[tag], datapaths[tag] = load_dataset_fron_config(**load_params, return_datapaths=True)
        print(f"Loaded {tag} with skeleton: {skeletons[tag].skeleton_name}")
        print("     Dataset size:", len(datasets[tag]))

    # prepare loaders
    dataloaders = {
        tag: DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for tag, dataset in datasets.items()
    }
    # dataset scale factors, placeholders only
    dataset_scales = {}
    for tag, dataset in datasets.items():
        if isinstance(dataset, torch.utils.data.Subset):
            dataset_scales[tag] = dataset.dataset.scale
        else:
            dataset_scales[tag] = dataset.scale
    return datasets, datapaths, skeletons, dataloaders, dataset_scales


def load_embeddings(exp):
    embeddings_path = os.path.join(exp, "analysis", "embeddings")
    embeddings = {}
    for name in sorted(os.listdir(embeddings_path)):
        embeddings[name.split('.')[0]] = torch.load(os.path.join(embeddings_path, name), weights_only=True)
    return embeddings


def compute_code_usage(codes, codebook_size, normalize=False):
    assert len(codes.shape) == 2
    
    if codes.shape[-1] == 2:
        codes_flattened = codes[:, 0] * codebook_size[1] + codes[:, 1]
    elif codes.shape[-1] == 1:
        codes_flattened = codes[:, 0]
    
    unique_codes, counts = np.unique(codes_flattened, return_counts=True)
    code_usage = np.zeros(np.prod(codebook_size), dtype=int)
    code_usage[unique_codes] = counts
    
    if normalize:
        code_usage = 100 * code_usage / code_usage.sum()

    code_usage = code_usage.reshape(*codebook_size)
    return code_usage


def intersection_over_union(a, b):
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b))
    return intersection / union


def get_datapaths(dataroot, datafile):
    if os.path.isdir(dataroot):
        candidates = [f for f in sorted(os.listdir(dataroot)) if not f.startswith('.')]
        if isinstance(datafile, list):
            datapaths = []
            for candidate in candidates:
                for file in datafile:
                    datapath = os.path.join(dataroot, candidate)
                    if os.path.isfile(datapath):
                        datapaths.append(datapath)
                    else:
                        datapaths.append(os.path.join(datapath, file))
        else:
            datapaths = [os.path.join(dataroot, candidate, datafile) for candidate in candidates] if datafile is not None else [os.path.join(dataroot, candidate) for candidate in candidates]
        datapaths = [dp for dp in datapaths if os.path.exists(dp)]
    elif os.path.isfile(dataroot):
        data = np.load(dataroot, allow_pickle=True)
        if dataroot.endswith('.npy'):
            data = data[()]
        datapaths = sorted(list(data.keys()))
    else:
        raise ValueError(f"Invalid dataroot: {dataroot}")

    return datapaths


def load_dataset_fron_config(
    dataset_name,
    config_relative_dir='../configs',
    filter_keys=[""],
    datafile="SDANNCE_x2/bsl0.5_FM/save_data_AVG0.mat",
    return_data_config_only=False,
    return_datapaths=False,
    embed_sliding=False,
    t_stride=None,
    overrides=None,
):
    assert os.path.exists(config_relative_dir), f"Config directory {config_relative_dir} does not exist"
    with initialize(config_path=config_relative_dir, version_base=None):
        available_configs = list(os.listdir(os.path.join(config_relative_dir, "dataset")))
        assert f"{dataset_name}.yaml" in available_configs, f"Dataset {dataset_name} not found in {available_configs}"
        
        cfg = compose(config_name=f"dataset/{dataset_name}.yaml").dataset
        del cfg.split
        dataset_config = deepcopy(cfg)
    
    if overrides is not None:
        for k, v in overrides.items():
            if hasattr(dataset_config, k):
                setattr(dataset_config, k, v)

    dataroot = dataset_config.dataroot
    if isinstance(dataroot, omegaconf.listconfig.ListConfig):
        datapaths = []
        for dr in dataroot:
            datapaths.extend(get_datapaths(dr, datafile))
    elif isinstance(dataroot, str):
        datapaths = get_datapaths(dataroot, datafile)
    else:
        raise ValueError(f"Invalid dataroot: {dataroot}")
    
    dataset_config.datapaths = sorted([datapaths[idx] for idx in filter_by_keys(filter_keys, datapaths)])
    print(f"Loading n={len(dataset_config.datapaths)} datafiles {dataset_config.datapaths}")

    # whether to apply sliding window during embedding
    if embed_sliding:
        assert t_stride is not None, "t_stride must be specified when embed_sliding is True"
        dataset_config['_target_'] = "csbev.dataset.base.WindowSlidingPoseDataset"
        dataset_config.t_stride = t_stride
    
    if return_data_config_only:
        return dataset_config

    dataset = instantiate(dataset_config)
    skeleton = dataset.skeleton
    
    if return_datapaths:
        return dataset, skeleton, dataset_config.datapaths
    
    return dataset, skeleton


def confidence_interval(a, confidence=0.95):
    return st.t.interval(confidence, len(a)-1, loc=np.mean(a, axis=0), scale=st.sem(a, axis=0))


def get_forehand_dist(kinematics, hand_indices):
    hand_diff_x = deepcopy(kinematics[:, :, hand_indices, 0])
    hand_diff_x = np.diff(hand_diff_x, axis=2)
    hand_diff_x *= 100
    return hand_diff_x[:, :, 0]


def get_hindlimb_dist(kinematics, foot_indices):
    foot_diff_x = deepcopy(kinematics[:, :, foot_indices, 0])
    foot_diff_x = np.diff(foot_diff_x, axis=2)
    foot_diff_x *= 100
    return foot_diff_x[:, :, 0]


def get_hindlimb_euclidean_dist(kinematics, foot_indices):
    foot_diff_x = deepcopy(kinematics[:, :, foot_indices])
    foot_diff_x = np.linalg.norm(np.diff(foot_diff_x, axis=2), axis=-1)
    return foot_diff_x[:, :, 0] * 100

def get_forehand_euclidean_dist(kinematics, hand_indices):
    hand_diff_x = deepcopy(kinematics[:, :, hand_indices])
    hand_diff_x = np.linalg.norm(np.diff(hand_diff_x, axis=2), axis=-1)
    return hand_diff_x[:, :, 0] * 100


def get_head_movement(kinematics, ear_indices, normalize=True):
    x = deepcopy(kinematics[:, :, ear_indices, 2])
    x = np.mean(x, axis=-1)
    t_start = x.shape[1] // 2
    if normalize:
        x -= x[:, t_start:t_start+1] # normalize by t = 0
    x *= 100
    return x


def find_match_codes(codes, code_idx, codebook_size):
    codes = codes.reshape(-1, 2)
    codes_flattened = codes[:, 0] * codebook_size[1] + codes[:, 1]

    code_idx = code_idx[0] * codebook_size[1] + code_idx[1]
    return np.where(codes_flattened == code_idx)[0]


def get_code_trajectories(
    kinematics,
    codes,
    code_idx,
    codebook_size,
    code_duration=16,
    t_span=16,
):
    code_indices = find_match_codes(codes, code_idx, codebook_size)
    if len(code_indices) == 0:
        return
    code_chunks = []
    for idx in code_indices:
        seq = kinematics[
            idx*code_duration - t_span : (idx+1)*code_duration + t_span
        ]
        if len(seq) < t_span*2 + code_duration:
            continue
        code_chunks.append(seq)
    return np.stack(code_chunks, axis=0) if len(code_chunks) > 0 else None


def get_common_keypoints(skeleton):
    if skeleton.skeleton_name == "rat23":
        handl_index = skeleton.keypoints.index("HandL")
        handr_index = skeleton.keypoints.index("HandR")
        earl_index = skeleton.keypoints.index("EarL")
        earr_index = skeleton.keypoints.index("EarR")
        footl_index = skeleton.keypoints.index("FootL")
        footr_index = skeleton.keypoints.index("FootR")
    elif skeleton.skeleton_name == "mouse22" or skeleton.skeleton_name == "pupdev":
        print(skeleton.skeleton_name)
        handl_index = skeleton.keypoints.index("ForepawL")
        handr_index = skeleton.keypoints.index("ForepawR")
        earl_index = skeleton.keypoints.index("EarL")
        earr_index = skeleton.keypoints.index("EarR")
        footl_index = skeleton.keypoints.index("HindpawL")
        footr_index = skeleton.keypoints.index("HindpawR")
    elif skeleton.skeleton_name == "human21":
        print(skeleton.skeleton_name)
        handl_index = skeleton.keypoints.index("LWrist")
        handr_index = skeleton.keypoints.index("RWrist")
        earl_index = skeleton.keypoints.index("LEar")
        earr_index = skeleton.keypoints.index("REar")
        footl_index = skeleton.keypoints.index("LHeel")
        footr_index = skeleton.keypoints.index("RHeel")
    elif skeleton.skeleton_name == "kpms_openfield_3d":
        print(skeleton.skeleton_name)
        handl_index = skeleton.keypoints.index("ForepawL")
        handr_index = skeleton.keypoints.index("ForepawR")
        earl_index = skeleton.keypoints.index("EarL")
        earr_index = skeleton.keypoints.index("EarR")
        footl_index = skeleton.keypoints.index("HindpawBaseL")
        footr_index = skeleton.keypoints.index("HindpawBaseR")
    else:
        raise ValueError(f"Unknown skeleton {skeleton.skeleton_name}")
    
    ear_indices = np.array([earl_index, earr_index])
    hand_indices = np.array([handl_index, handr_index])
    foot_indices = np.array([footl_index, footr_index])
    return ear_indices, hand_indices, foot_indices


def align_kinematics_by_code(
    kinematics,
    codes,
    code_idx,
    codebook_size,
    skeleton,
    tag,
    y_offset=0,
    code_duration=16,
    t_span=16,
    figsize=(40, 40),
    ylim=(-35, 35),
    color='darkred',
    color_alpha=0.3,
    stats_type = "forehand",
    normalize_at_onset=False,
    threshold=100,
    ax=None,
    fig=None,
    add_title=True,
    save_video=False,
    n_videos=5,
    save_images=False,
    savename=None,
    plot_dir=None,
    **kwargs
):
    hand_indices, ear_indices, foot_indices = get_common_keypoints(skeleton)

    code_indices = find_match_codes(codes, code_idx, codebook_size)
    if len(code_indices) == 0:
        return None, None
    code_chunks = []
    for idx in code_indices:
        seq = kinematics[
            idx*code_duration - t_span : (idx+1)*code_duration + t_span
        ]
        if len(seq) < t_span*2 + code_duration:
            continue
        code_chunks.append(seq)

    code_chunks = np.stack(code_chunks, axis=0)
    print(f"{code_idx}: {code_chunks.shape}")

    if stats_type == "forehand":
        stats = get_forehand_dist(code_chunks, hand_indices)
    elif stats_type == "hindlimb":
        stats = get_hindlimb_dist(code_chunks, foot_indices)
    elif stats_type == "head":
        stats = get_head_movement(code_chunks, ear_indices)
    elif stats_type == "hindlimb_euclidean":
        stats = get_hindlimb_euclidean_dist(code_chunks, foot_indices)
    elif stats_type == "forehand_euclidean":
        stats = get_forehand_euclidean_dist(code_chunks, hand_indices)

    if "mouse" in tag:
        stats /= 2.0

    if ax is None:
        reset_rcparams()
        major = 1
        plt.rcParams["xtick.major.size"] = major
        plt.rcParams["ytick.major.size"] = major
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0]*mm, figsize[1]*mm), dpi=300)
        hide_axes_top_right(ax)

    # plot
    stats_mean = stats.mean(axis=0)
    stats_center = stats_mean[len(stats_mean)//2]
    conf95 = confidence_interval(stats, confidence=0.95)
    
    if y_offset != 0:
        stats_mean = stats_mean - stats_center + y_offset
        conf95 = conf95 - stats_center + y_offset
    
    ax.plot(np.arange(t_span*2+code_duration)-4-t_span, stats_mean, linewidth=0.8, color=color)

    ax.fill_between(
        np.arange(t_span*2+code_duration)-4-t_span, conf95[0], conf95[1],
        color=color, alpha=color_alpha,
        linewidth=0.0,
    )

    # ax.set_ylabel("Forepaw x distance (L-R, mm)")
    # ax.set_xlabel("Frame relative to motif center")
    code_frac = 100 * len(code_indices) / codes.shape[0]
    ax.set_xticks([-t_span-code_duration//2, 0, t_span+code_duration//2])
    
    if add_title:
        ax.set_title(f"Code{code_idx} ({code_frac:.2f}%)", fontsize=6)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if plot_dir is not None and savename is not None:
        if save_video:
            video_dir = os.path.join(plot_dir, "code_videos", savename)
            if not os.path.exists(video_dir): os.makedirs(video_dir)

            for i, idx in enumerate(np.random.choice(len(code_chunks), n_videos, replace=False)):
                # fig = plt.figure(figsize=(40*mm, 40*mm), dpi=300)
                # ax = fig.add_subplot(111, projection='3d')
                poseseq = code_chunks[idx]
                poseseq = poseseq - poseseq.mean(axis=(0,1), keepdims=True)
                make_pose3d_seq_video(
                    poseseq,
                    skeleton,
                    savename=os.path.join(video_dir, f"{tag}_{i}.gif"),
                    fps=30,
                    linewidth=0.5,
                    # lineborder=0.,
                    marker_size=10,
                    marker_edgewidth=0,
                    marker_linewidth=0.2,
                    coord_limits=0.75,
                    colormap=plt.get_cmap("Spectral", 6),
                    colors_chain='k',
                )
        if save_images:
            fig.savefig(
                os.path.join(plot_dir, f"{stats_type}_fill_{savename}.pdf"),
                bbox_inches='tight',
                transparent=True,
            )
    
    return fig, ax


def vis_latent_space(exp):
    latent_space_dir = os.path.join(exp, "analysis", "latent_space")
    dataset_tags = sorted(os.listdir(latent_space_dir))
    latent_spaces = {
        dataset_tag: plt.imread(os.path.join(latent_space_dir, dataset_tag))
        for dataset_tag in dataset_tags
    }
    reset_rcparams()
    fig, axes = plt.subplots(1, len(latent_spaces), figsize=(5*len(latent_spaces), 5), dpi=200)
    axes = axes.flatten() if len(latent_spaces) > 1 else [axes]
    for ax, (dataset_tag, latent_space) in zip(axes, latent_spaces.items()):
        ax.imshow(latent_space)
        ax.set_title(dataset_tag)
        hide_axes_top_right(ax)
        hide_axes_all(ax)


def automatic_sort_motifs(exp, anchor_idx=0, codebook_idx=0):
    ckpt_path = os.path.join(exp, "checkpoints", "model.pth")
    ckpt = torch.load(ckpt_path, weights_only=False)
    codebook_vecs = {
        k: ckpt["model"][k].clone().detach().cpu()
        for k in ckpt["model"].keys() if "bottleneck" in k
    }
    
    codebook = list(codebook_vecs.values())[codebook_idx]
    anchor = F.normalize(codebook[anchor_idx].unsqueeze(0), dim=1)
    similarity = F.normalize(codebook, dim=1) @ anchor.T
    similarity = similarity.squeeze().numpy()
    sorting_indices = np.argsort(similarity)[::-1]
    print(list(codebook_vecs.keys())[codebook_idx], sorting_indices)
    return list(sorting_indices)


def plot_ethogram(data, start, end, distinct_syllables, code_span=16, figsize=(60, 40), aspect=0.3, annot_axis=True, save_path=None):
    reset_rcparams()
    plt.rcParams['axes.linewidth'] = 0.3
    plt.rcParams['ytick.major.width'] = 0.0
    plt.rcParams['ytick.major.pad'] = 1
    plt.rcParams['ytick.labelsize'] = 4
    figsize = (figsize[0]*mm, figsize[1]*mm)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)

    start_idx = start // code_span
    end_idx = end // code_span
    codes = data[start_idx:end_idx, 0]
    codes = codes.cpu().numpy()
    t_duration = len(codes)
    cmap = plt.get_cmap("tab20b", len(distinct_syllables))

    # expand codes into 2d ethogram
    ethogram_expanded = np.zeros((t_duration, len(distinct_syllables), 4))
    for t, syllable in enumerate(codes):
        if syllable not in distinct_syllables:
            continue
        syllable_index = distinct_syllables.index(syllable)
        ethogram_expanded[t, syllable_index] = cmap(syllable_index)
    ethogram_expanded = ethogram_expanded.transpose(1, 0, 2)

    # plot
    im = ax.imshow(ethogram_expanded, interpolation="nearest", aspect=aspect*(end_idx-start_idx)/len(distinct_syllables))
    ax.set_xlim([0, end_idx-start_idx])
    
    if annot_axis:
        ax.set_xlabel("Time (s)")
        xticks = np.arange(0, end_idx-start_idx+1, 500)
        print(xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks * code_span / 50)
        ax.set_yticks(np.arange(len(distinct_syllables)))
        ax.set_yticklabels(distinct_syllables)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # add a scale bar of 10 seconds
    # ax.axhline(y=8.0, xmin=0, xmax=1/(end-start)*10*50, color='k', linestyle='-', linewidth=0.5)
    
    if save_path is not None:
        savedir = os.path.dirname(save_path)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        fig.savefig(save_path, bbox_inches="tight", transparent=True)
    return fig, ax


def interpolate_along_axis(x, xp, fp, axis=0):
    assert len(xp.shape) == len(x.shape) == 1
    assert fp.shape[axis] == len(xp)
    assert len(xp) > 0, "xp must be non-empty; cannot interpolate without datapoints"

    fp = np.moveaxis(fp, axis, 0)
    shape = fp.shape[1:]
    fp = fp.reshape(fp.shape[0], -1)

    x_interp = np.zeros((len(x), fp.shape[1]))
    for i in range(fp.shape[1]):
        x_interp[:, i] = np.interp(x, xp, fp[:, i])
    x_interp = x_interp.reshape(len(x), *shape)
    x_interp = np.moveaxis(x_interp, 0, axis)
    return x_interp


def normalize_poseseq(
    poseseq,
    normalize_method: Literal['grounding', 'centering', 'spine', 'none'] = 'grounding',
    scale=1.0,
):
    if not isinstance(poseseq, np.ndarray):
        poseseq = poseseq.clone().detach().cpu().numpy()
    else:
        poseseq = deepcopy(poseseq)
        
    if 'grounding' == normalize_method:
        poseseq -= np.min(poseseq[:, :, 2:], axis=1, keepdims=True)
    elif 'centering' == normalize_method:
        poseseq -= np.mean(poseseq, axis=1, keepdims=True)

    return poseseq * scale  # scale the pose_seq if needed


def plot_pose_sequence_with_interpolation(
    pose_seq,
    skeleton,
    n_samples=3,
    colormap='RdYlBu',
    colors_chain=None,
    colors_marker=None,
    linewidth=0.5,
    marker_size=2,
    marker_edge_width=0.5,
    marker_line_width=0.3,
    figsize=(50, 30),
    offset=1,
    xlim_offset=1,
    ylim=(0.1, 4.9),
    interpolation_n=500,
    interpolation_alpha=0.5,
    savename=None,
    figure_dir=None,
    coords=(0, 2),
    aspect=1.0,
):
    reset_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=(figsize[0]*mm, figsize[1]*mm), dpi=300)
    # datadim = pose_seq.shape[-1]
    # if datadim == 2:
    #     fig, ax = plt.subplots(1, 1, figsize=(figsize[0]*mm, figsize[1]*mm), dpi=300)
    # elif datadim == 3:
    #     fig = plt.figure(figsize=(figsize[0]*mm, figsize[1]*mm), dpi=300)
    #     ax = fig.add_subplot(111, projection='3d')
    # else:
    #     raise ValueError(f"Invalid pose_seq datadim {datadim}. Must be 2 or 3.")
    hide_axes_all(ax)

    # skeleton related
    kinematic_tree = skeleton.kinematic_tree_indices
    body_region_indices = skeleton.body_region_indices
    n_body_regions = len(skeleton.body_regions)
    n_chains = len(kinematic_tree)
    unique_markers = np.unique(sum(kinematic_tree, []))

    if colormap is not None:
        if isinstance(colormap, str):
            colormap = plt.get_cmap(colormap)
        colors_chain = [colormap(i / n_chains) for i in range(n_chains)]
        colors_chain = np.array(colors_chain)
        colors_marker = np.array([colormap(i / n_body_regions) for i in body_region_indices])
    else:
        assert colors_marker is not None

    n_frames = pose_seq.shape[0]
    frames_to_plot = np.linspace(0, n_frames - 1, n_samples, dtype=int)
    # pose_seq[:, :, 0] -= pose_seq[0, 0, 0]
    data_min, data_max = pose_seq[:, :, coords[0]].min(), pose_seq[:, :, coords[0]].max()
    for t in range(n_frames):
        pose = deepcopy(pose_seq[t])
        pose = pose[:, [coords[0], coords[1]]]
        pose[:, 0] += offset*t
        
        data_max = max(data_max, pose[:, 0].max())

        # plot explicitly on key frames
        if t in frames_to_plot:
            for chain, color in zip(kinematic_tree, colors_chain):
                ax.plot(*pose[chain].T,
                    linewidth=linewidth,
                    color='k',
                    # color=color,
                    zorder=2,
                )
                ax.scatter(
                    *pose[unique_markers].T,
                    c=colors_marker[unique_markers], 
                    s=marker_size+marker_edge_width,
                    zorder=4,
                    edgecolor="k",
                    # edgecolor=colors_marker[unique_markers],
                    linewidth=marker_line_width, 
                )
        
    # display intermediate frames
    for marker in unique_markers:
        x = pose_seq[:, marker, 0] + offset*np.arange(n_frames)
        y = pose_seq[:, marker, 2:]
        x_interp = np.linspace(x[0], x[-1], interpolation_n)
        y_interp = interpolate_along_axis(x_interp, x, y, axis=0)[:, 0]
        ax.plot(x_interp, y_interp, color=colors_marker[marker], linewidth=linewidth, zorder=1, alpha=interpolation_alpha)
        
    # ax.set_aspect(aspect / (ylim[1] - ylim[0]) * (data_max - data_min+2*xlim_offset))
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(ylim)
    ax.set_xlim([data_min-xlim_offset, data_max+xlim_offset])
    plt.tight_layout(pad=0.0)

    if savename is not None and figure_dir is not None:
        fig.savefig(
            os.path.join(figure_dir, f"{savename}.pdf"),
            # bbox_inches='tight',
            transparent=True,
    )
    return fig, ax


def load_experiment(exproot):
    assert os.path.exists(exproot), f"Experiment {exproot} does not exist"

    # load config
    config_path = os.path.join(exproot, "config.yaml")
    config = OmegaConf.load(config_path)

    # skeleton profiles available from training
    dataset_tags = list(config.datasets.keys())
    skeletons = {
        dataset_tag: instantiate(config.datasets[dataset_tag].skeleton)
        for dataset_tag in dataset_tags
    }
    print("Avilable skeletons:", skeletons.keys())

    # load model checkpoint
    ckpt_path = os.path.join(exproot, "checkpoints", "model.pth")
    ckpt = torch.load(ckpt_path, weights_only=False)

    # load model
    model = instantiate(config.model)
    model.load_state_dict(ckpt["model"])
    model = model.eval().cuda()
    print(f"Model loaded from Epoch {ckpt['metadata']['epoch']}")

    print("All codebooks:")
    codebook_vecs = {
        k: ckpt["model"][k].clone().detach().cpu()
        for k in ckpt["model"].keys() if "bottleneck" in k
    }
    for k, cb in codebook_vecs.items():
        print(k, cb.shape)

    codebook_size = [cb.shape[0] for cb in codebook_vecs.values()]
    if config.model.bottleneck.sub_quantizer._target_ == 'csbev.model.quantizer.ResidualVQ':
        sub_quantizer_depth = config.model.bottleneck.sub_quantizer.num_quantizers
        codebook_size = codebook_size[::sub_quantizer_depth]

    assert codebook_size == config.model.bottleneck.codebook_size
    print(f"Codebook size: {codebook_size}")
    
    if hasattr(config.model.encoder, "channel_encoder"):
        n_ds = config.model.encoder.channel_encoder.n_ds
    else:
        n_ds = config.model.encoder.encoder_shared.n_ds
    
    code_duration = 2 ** n_ds
    print(f"Code duration: {code_duration}")
    
    return config, model, skeletons, codebook_vecs, code_duration, codebook_size


def decode_motif_kinematics(model, skeleton, codebook_size, output_tag):
    _, traj, combos = model.sample_latent_codes(output_tag, repeats=1)
    traj = traj.reshape(*codebook_size, *traj.shape[1:])
    traj = traj.reshape(
    *traj.shape[:2], skeleton.n_keypoints, skeleton.datadim, -1
    )
    traj = traj.permute(0, 1, 4, 2, 3)
    traj = traj.flatten(0, 1)
    traj = traj.detach().cpu().numpy()
    print(f"Decoded kinematics ({skeleton.skeleton_name}): {traj.shape}")
    
    return {combo: x for combo, x in zip(combos, traj)}


def plot_motif_with_interpolation(
    motif,
    skeleton,
    interpolation_num_points=100,
    n_samples=2,
    alpha=1.0,
    colormap='RdYlBu',
    colors_chain=None,
    colors_marker=None,
    linewidth: float = 2.0,
    lineborder: float = 0.5,
    lineborder_color: str = "k",
    marker_size: float = 30,
    marker_edgewidth: float = 2,
    marker_edgecolor: str = "k",
    marker_linewidth: float = 0.5,
    coord_limits: float = 1.0,
    alpha_start=0.3,
    alpha_end=1.0,
    figsize=(30,30),
    view_init=(45, -60, 0),
    fig=None,
    ax=None,
    savename=None,
    figure_dir=None,
):
    reset_rcparams()
    if fig is None or ax is None:
        fig = plt.figure(figsize=(figsize[0] * mm, figsize[1] * mm), dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=True)
    
    # adjust view
    ax.view_init(elev=view_init[0], azim=view_init[1], roll=view_init[2])
    
    hide_axes_all(ax)
    
    # interpolate the pose sequence
    n_frames_original = motif.shape[0]
    interpolated_keypoints = [
        smoothly_interpolate_3d_points(motif[:, i], num_points=interpolation_num_points)
        for i in range(motif.shape[1])
    ]
    interpolated_motif = np.stack(interpolated_keypoints, axis=1)
    pose_seq = interpolated_motif
    
    # skeleton related
    kinematic_tree = skeleton.kinematic_tree_indices
    body_region_indices = skeleton.body_region_indices
    n_body_regions = len(skeleton.body_regions)
    n_chains = len(kinematic_tree)
    unique_markers = np.unique(sum(kinematic_tree, []))

    if colormap is not None:
        if isinstance(colormap, str):
            colormap = plt.get_cmap(colormap)
        colors_marker = np.array([colormap(i / n_body_regions) for i in body_region_indices])
    else:
        assert colors_marker is not None  

    if colors_chain is None:
        colors_chain = [colormap(i / n_chains) for i in range(n_chains)]
        colors_chain = np.array(colors_chain)
    else:
        colors_chain = [colors_chain] * n_chains

    # plot interpolation
    for marker in unique_markers:
        traj = pose_seq[:, marker]
        ax.plot3D(*traj.T, linewidth=linewidth, zorder=1, alpha=(alpha_start + alpha_end) / 3, color=colors_marker[marker], linestyle='-')

    n_frames = pose_seq.shape[0]
    frames_to_plot = np.linspace(0, n_frames - 1, n_samples, dtype=int)
    alphas = np.linspace(alpha_start, alpha_end, n_samples)
    
    # print(frames_to_plot)
    for relative_idx, (t, alpha) in enumerate(zip(frames_to_plot, alphas)):
        pose = pose_seq[t]
        # plot explicitly on key frames
        for chain, color in zip(kinematic_tree, colors_chain):
            ax.plot3D(
                *pose[chain].T, color=color, zorder=3,
                alpha=alpha, linewidth=linewidth,
            )
            if t != 0:
                ax.plot3D(
                    *pose[chain].T,
                    color=lineborder_color,
                    zorder=2,
                    alpha=alpha,
                    linewidth=linewidth + lineborder,
                )
        # Plot nodes (keypoints)
        ax.scatter(
            *pose.T, c=colors_marker, s=marker_size,
            zorder=5,
            alpha=alpha,
            linewidth=0,
        )
        ax.scatter(
            *pose.T,
            c=colors_marker,
            s=marker_size + marker_edgewidth,
            zorder=4,
            alpha=alpha,
            edgecolor=marker_edgecolor,
            linewidth=marker_linewidth * 0.2 if t == 0 else marker_linewidth,
        )

    if isinstance(coord_limits, (int, float)):
        ax.set_xlim([-coord_limits, coord_limits])
        ax.set_ylim([-coord_limits, coord_limits])
        ax.set_zlim([-coord_limits, coord_limits])
    # else:
    #     ax.set_xlim(coord_limits)
    #     ax.set_ylim(coord_limits)
    #     ax.set_zlim(coord_limits)
    ax.set_axis_off()

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    if savename is not None and figure_dir is not None:
        fig.savefig(
            os.path.join(figure_dir, f"{savename}.pdf"),
            # bbox_inches='tight',
            transparent=True,
    )
    return fig, ax


def l2_dist(x, y):
    return np.sqrt(np.sum((x - y)**2, axis=-1))


def angle_between(v1, v2):
    assert v1.shape == v2.shape, "Vectors must have the same shape"
    data_shape = v1.shape[:-1]
    v1 = v1.reshape(-1, v1.shape[-1])
    v2 = v2.reshape(-1, v2.shape[-1])
    
    cos_theta = np.sum(v1*v2, axis=1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    ang = np.arccos(cos_theta) / np.pi * 180  # convert to degrees and shift to [-90, 90]
    ang = ang.reshape(data_shape)
    return ang


def angle_between_signed(v, ref):
    cos_theta = np.sum(v*ref, axis=1) / (np.linalg.norm(v, axis=-1) * np.linalg.norm(ref, axis=-1))
    return cos_theta


def compute_kinematic_attributes(
    motif_kinematics,
    skeleton,
    spineF_keypoint='SpineF',
    spineL_keypoint='SpineL',
    head_keypoint='Snout',
    tail_keypoint='TailBase',
    forelimb_keypoints=['HandL', 'HandR'],
    hindlimb_keypoints=['FootL', 'FootR'],
):
    n_samples, T, n_kpts, datadim = motif_kinematics.shape
    assert n_kpts == len(skeleton.keypoints)

    head_idx = skeleton.keypoints.index(head_keypoint)
    tail_idx = skeleton.keypoints.index(tail_keypoint)
    spineF_idx = skeleton.keypoints.index(spineF_keypoint)
    spineL_idx = skeleton.keypoints.index(spineL_keypoint)
    forelimb_idx = [skeleton.keypoints.index(kpt) for kpt in forelimb_keypoints]
    hindlimb_idx = [skeleton.keypoints.index(kpt) for kpt in hindlimb_keypoints]
    
    # velocity
    velocity = np.linalg.norm(motif_kinematics[:, 1:, :] - motif_kinematics[:, :-1, :], axis=-1)
    velocity_avg = np.mean(velocity, axis=1) # average over time
    velocity_by_parts = {}
    for br_idx, body_region in enumerate(skeleton.body_regions):
        body_region_indices = np.where(skeleton.body_region_indices == br_idx)[0]
        velocity_by_parts[body_region] = np.mean(velocity[:, :, body_region_indices], axis=-1)
    
    # velocity by direction (x, y, z)
    velocity_by_direction = motif_kinematics[:, 1:, :, :] - motif_kinematics[:, :-1, :, :]  # (n_samples, T-1, n_kpts, datadim)
    
    # velocity in xy
    velocity_xy = np.linalg.norm(
        velocity_by_direction[:, :, :, :2], axis=-1
    )
    
    # distance between head and tail
    dist_head_tail = np.linalg.norm(motif_kinematics[:, :, head_idx] - motif_kinematics[:, :, tail_idx], axis=-1)
    
    # head-body angle in xy
    head_vector = motif_kinematics[:, :, head_idx] - motif_kinematics[:, :, spineF_idx]
    tail_vector = motif_kinematics[:, :, tail_idx] - motif_kinematics[:, :, spineL_idx]
    body_vector = motif_kinematics[:, :, spineF_idx] - motif_kinematics[:, :, spineL_idx]
    
    alignment_vector = np.array([0, 1, 0]) # +y
    alignment_vector = np.tile(alignment_vector, (*head_vector.shape[:-1], 1))
    
    # > 0 left, < 0 right
    head_angle = 90 - angle_between(alignment_vector[..., :2], head_vector[..., :2])
    tail_angle = 90 - angle_between(alignment_vector[..., :2], tail_vector[..., :2])
    
    head_tail_angle = angle_between(head_vector[..., :2], tail_vector[..., :2])
    head_body_angle = angle_between(head_vector[..., :2], body_vector[..., :2])
    tail_body_angle = angle_between(tail_vector[..., :2], body_vector[..., :2])
    
    # head angle relative to horizontal
    alignment_z_vector = np.array([0, 0, 1])
    alignment_z_vector = np.tile(alignment_z_vector, (*head_vector.shape[:-1], 1))
    head_angle_z = 90 - angle_between(alignment_z_vector[..., [0, 2]], head_vector[..., [0, 2]])
    head_body_angle_z = angle_between(head_vector[..., [0, 2]], body_vector[..., [0, 2]]) - 90

    # body z height
    zheight = np.max(motif_kinematics[:, :, :, 2], axis=2) - np.min(motif_kinematics[:, :, :, 2], axis=2)
    
    # forelimb movements
    dist_forelimb = motif_kinematics[:, :, forelimb_idx[0]] - motif_kinematics[:, :, forelimb_idx[1]]
    dist_hindlimb = motif_kinematics[:, :, hindlimb_idx[0]] - motif_kinematics[:, :, hindlimb_idx[1]]
    dist_forelimb = dist_forelimb[..., 0]
    dist_hindlimb = dist_hindlimb[..., 0]
    
    results = {
        "velocity": velocity,
        "velocity_by_keypoint": velocity_avg,
        "velocity_by_region": velocity_by_parts,
        "velocity_by_direction": velocity_by_direction,
        "velocity_xy": velocity_xy,
        "dist_head_tail": dist_head_tail,
        "head_angle": head_angle,
        "tail_angle": tail_angle,
        "head_tail_angle": head_tail_angle,
        "head_body_angle": head_body_angle,
        "tail_body_angle": tail_body_angle,
        "head_z_angle": head_angle_z,
        "head_body_z_angle": head_body_angle_z,
        "zheight": zheight,
        "dist_forelimb": dist_forelimb,
        "dist_hindlimb": dist_hindlimb,
    }
    return results


def plot_attribute_value_bar_plot(
    attribute_values,
    codebook_size,
    codes_sorting=None,
    cb1_colors=None,
    figsize=(120, 15),
    aspect=0.05,
    cmap='turbo',
    relaxation=0.1,
    bar_alpha=0.9,
    plot_shaded_region=True,
    shade_same_color=None,
    background_alpha=0.5,
    figure_dir=None,
    savename=None,
):
    data_to_plot = attribute_values.copy()
    if codes_sorting is not None:
        data_to_plot = data_to_plot[codes_sorting]
    
    data_min, data_max = data_to_plot.min(), data_to_plot.max()
    data_min = data_min * (1 - np.sign(data_min) * relaxation)
    data_max = data_max * (1 + relaxation)
    
    # category colors
    if cb1_colors is None:
        cb1_cmap = plt.get_cmap(cmap, codebook_size[0])
        cb1_colors = [cb1_cmap(i) for i in range(codebook_size[0])]
        cb1_colors = np.repeat(cb1_colors, codebook_size[1], axis=0)
        assert len(cb1_colors) == attribute_values.shape[0]
    
        if codes_sorting is not None:
            cb1_colors = cb1_colors[codes_sorting]
    
    reset_rcparams()
    fig, ax = plt.subplots(1, 1, figsize=get_figsize_in_mm(figsize), dpi=300)
    
    x = np.arange(data_to_plot.shape[0])
    ax.bar(
        x,
        data_to_plot,
        width=0.95,
        edgecolor='w',
        linewidth=0.1,
        alpha=bar_alpha,
        color=cb1_colors,
        zorder=1,
    )
    ax.set_ylim([data_min, data_max])
    ax.set_xlim([-0.5, len(data_to_plot) - 0.5])  # ensure the bars are centered

    if plot_shaded_region:
        if shade_same_color is not None:
            ax.axvspan(
                -0.5, len(data_to_plot) - 0.5,
                color=shade_same_color,
                alpha=background_alpha,
                linewidth=0,
                zorder=-1,
            )
        else:
            # draw colored regions behind the bars
            for i in range(codebook_size[0]):
                start = i * codebook_size[1]
                end = (i + 1) * codebook_size[1]
                ax.axvspan(start-0.5, end-0.5, 0, 1, color=cb1_colors[start], alpha=background_alpha, linewidth=0, zorder=-1)

    ax.set_aspect(aspect / (data_max - data_min) * np.prod(codebook_size))
    ax.set_axis_off()
    plt.tight_layout(pad=0.0)
    
    if figure_dir is not None and savename is not None:
        fig.savefig(
            os.path.join(figure_dir, savename),
            bbox_inches='tight',
            transparent=True,
        )
    return fig, ax


def plot_usage_nonzero_only_2d(
    data,
    is_diff=False,
    codebook1_sorting=None,
    codes_all_sorting=None,
    threshold=0.5,
    aspect=0.25,
    axes_linewidth=0.3,
    figsize=(60, 20), # in mm
    cmap='BuGn', # colormap for imshow
):
    reset_rcparams()
    plt.rcParams['axes.linewidth'] = axes_linewidth
    plt.rcParams['ytick.major.size'] = 0

    fig, ax = plt.subplots(1, 1, figsize=get_figsize_in_mm(figsize), dpi=300)
    to_plot = data.copy() #[codebook_size1, codebook_size2]
    
    if codebook1_sorting is not None:
        to_plot = to_plot[codebook1_sorting, :]
    
    data_max = to_plot.max()
    data_min = to_plot.min()
    data_extreme = max(abs(data_min), abs(data_max))
    
    cmap = plt.get_cmap(cmap)
    
    cb1_size, cb2_size = to_plot.shape
    counts = [0] * cb1_size 
    for i in range(cb1_size):
        cnt = 0
        for j in range(cb2_size):
            if to_plot[i, j] <= threshold:
                continue
            x, y = j, cb1_size - 1 - i
            patch = plt.Rectangle(
                (x, y), 1, 1,
                facecolor=cmap((to_plot[i, j] / data_extreme)),
                edgecolor='k',
                linewidth=0.1,
            )
            ax.add_patch(patch)
            cnt += 1
        counts[i] = cnt
    
    ax.set_xlim(0, to_plot.shape[1])
    ax.set_ylim(0, to_plot.shape[0])
    ax.set_aspect(aspect)
    # ax.set_axis_off()
    plt.tight_layout(pad=0.0)
    return fig, ax, counts


def plot_usage(
    data,
    codebook_size,
    is_diff=False,
    codebook1_sorting=None,
    codes_all_sorting=None,
    aspect=0.25,
    axes_linewidth=0.3,
    figsize=(60, 30), # in mm
    cmap='BuGn', # colormap for imshow
    annot_text=False,
    show_xticks=False,
    show_subgroups=False,
    show_colorbar=True,
    yticklabels=[],
    cbar_params={'fraction': 0.01, 'pad': 0.04}, # colorbar params
):
    reset_rcparams()
    plt.rcParams['axes.linewidth'] = axes_linewidth
    plt.rcParams['ytick.major.size'] = 0

    fig, ax = plt.subplots(1, 1, figsize=get_figsize_in_mm(figsize), dpi=300)
    to_plot = data.copy() #[n_exps, codebook_size1, codebook_size2]
    
    if codebook1_sorting is not None:
        to_plot = to_plot[:, codebook1_sorting, :]
    
    to_plot = to_plot.reshape((to_plot.shape[0], -1))
    if codes_all_sorting is not None:
        to_plot = to_plot[:, codes_all_sorting]

    data_max = np.ceil(to_plot.max())
    data_min = np.floor(to_plot.min())
    data_extreme = max(abs(data_min), abs(data_max))
    
    im = ax.imshow(
        to_plot,
        cmap=cmap,
        aspect=aspect * to_plot.shape[1] / to_plot.shape[0],
        interpolation='nearest',
        vmin=0 if not is_diff else -data_extreme,  # for diff plots, allow negative values
        vmax=data_max if not is_diff else data_extreme,  # for diff plots, allow positive values
    )
    ax.set_yticks([]) if not yticklabels else ax.set_yticks(np.arange(len(yticklabels))) # remove y ticks if no labels provided
    if yticklabels:
        ax.set_yticklabels(yticklabels, fontsize=5)
    
    xticks = np.arange(0, to_plot.shape[1], codebook_size[1]) if show_xticks else []
    ax.set_xticks(xticks)
    if show_xticks and codebook1_sorting is not None:
        xtick_labels = xticks[codebook1_sorting] // codebook_size[1]  # map back to original codebook indices
        ax.set_xticklabels(xtick_labels)
    
    if show_subgroups:
        for x in np.arange(0, to_plot.shape[1], codebook_size[1])[1:]:
            ax.axvline(x=x, color='k', linestyle='--', linewidth=axes_linewidth)
    
    if annot_text:
        for i in range(to_plot.shape[0]):
            for j in range(to_plot.shape[1]):
                # Annotate each cell with its value
                val = to_plot[i, j]
                if val > 0:
                    ax.text(j, i, f"{int(val)}", ha='center', va='center', fontsize=5, color='black')
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, **cbar_params)
        cbar_ylim = cbar.ax.get_ylim()
        cbar.set_ticks([np.floor(cbar_ylim[0]), np.ceil(cbar_ylim[-1])])

    return fig, ax, to_plot