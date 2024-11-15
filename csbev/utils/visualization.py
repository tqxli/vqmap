from __future__ import annotations
from typing import List
import os
import hydra
import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from loguru import logger
import torch
import tqdm

from csbev.dataset.skeleton import SkeletonProfile


mm = 1 / 25.4
plt.rcParams["font.family"] = "arial"


def reset_rcparams():
    matplotlib.rcdefaults()
    matplotlib.rcParams["font.family"] = "Arial"
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["lines.linewidth"] = 0.8
    plt.rcParams["axes.labelpad"] = 2

    major = 2
    majorpad = 2
    plt.rcParams["xtick.major.size"] = major
    plt.rcParams["ytick.major.size"] = major
    plt.rcParams["xtick.major.width"] = 0.4
    plt.rcParams["ytick.major.width"] = 0.4
    plt.rcParams["xtick.major.pad"] = majorpad
    plt.rcParams["ytick.major.pad"] = majorpad
    plt.rcParams["xtick.labelsize"] = 5
    plt.rcParams["ytick.labelsize"] = 5
    plt.rcParams["axes.labelsize"] = 5

    plt.rcParams["legend.borderpad"] = 0
    plt.rcParams["legend.framealpha"] = 0
    plt.rcParams["legend.fontsize"] = 5


def hide_axes_top_right(ax):
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)


def hide_axes_all(ax):
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)


def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_pose2d(
    pose: np.ndarray,
    skeleton: SkeletonProfile,
    ax: matplotlib.axes.Axes = None,
    figsize: tuple = (80 * mm, 80 * mm),
    alpha: float = 1.0,
    colors_chain: List[str] | None = None,
    colors_marker: List[str] | None = None,
    colormap: str = "RdYlGn",
    linewidth: float = 2.0,
    lineborder: float = 0.5,
    lineborder_color: str = "k",
    marker_size: float = 30,
    marker_edgewidth: float = 2,
    marker_edgecolor: str = "k",
    coord_limits: float = 1.,  
):
    """Plot a single 2D pose.

    Args:
        pose (np.ndarray): a single pose of shape [n_keypoints, 2 or 3]
        skeleton (SkeletonProfile): class containing the skeleton information
    """
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111)

    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    assert (
        pose.shape[0] == skeleton.n_keypoints
    ), f"Pose shape {pose.shape} does not match the skeleton {skeleton.n_keypoints}"
    assert pose.shape[1] == 2, f"Pose shape {pose.shape} should be [n_keypoints, 2]"

    n_keypoints = skeleton.n_keypoints
    kinematic_tree = skeleton.kinematic_tree_indices
    body_region_indices = skeleton.body_region_indices
    n_body_regions = len(skeleton.body_regions)
    n_chains = len(kinematic_tree)

    colormap = plt.get_cmap(colormap)
    if colors_chain is None:
        colors_chain = [colormap(i / n_chains) for i in range(n_chains)]
    elif isinstance(colors_chain, str):
        colors_chain = [colors_chain] * n_chains
    colors_chain = np.array(colors_chain)

    if colors_marker is None:
        colors_marker = np.array([colormap(i / n_body_regions) for i in body_region_indices])

    assert (
        len(colors_chain) == n_chains and len(colors_marker) == n_keypoints
    ), "Color list should match the number of kinematic chains and keypoints"

    # Plot edges (segments)
    for chain, color in zip(kinematic_tree, colors_chain):
        ax.plot(
            *pose[chain].T,
            color=lineborder_color,
            zorder=2,
            alpha=alpha,
            linewidth=linewidth + lineborder,
        )
        ax.plot(
            *pose[chain].T, color=color, zorder=3, alpha=alpha, linewidth=linewidth,
        )
    # Plot nodes (keypoints)
    unique_markers = np.unique(sum(kinematic_tree, []))
    ax.scatter(
        *pose[unique_markers].T, c=colors_marker[unique_markers], s=marker_size, zorder=1, alpha=alpha, linewidth=0,
    )
    ax.scatter(
        *pose[unique_markers].T,
        c=colors_marker[unique_markers],
        s=marker_size + marker_edgewidth,
        zorder=4,
        alpha=alpha,
        edgecolor=marker_edgecolor,
        linewidth=0.5,
    )

    ax.set_xlim([-coord_limits, coord_limits])
    ax.set_ylim([-coord_limits, coord_limits])
    ax.set_axis_off()


def visualize_pose3d(
    pose: np.ndarray,
    skeleton: SkeletonProfile,
    ax: matplotlib.axes.Axes = None,
    figsize: tuple = (80 * mm, 80 * mm),
    alpha: float = 1.0,
    colors_chain: List[str] | None = None,
    colors_marker: List[str] | None = None,
    colormap: str = "RdYlGn",
    linewidth: float = 2.0,
    lineborder: float = 0.5,
    lineborder_color: str = "k",
    marker_size: float = 30,
    marker_edgewidth: float = 2,
    marker_edgecolor: str = "k",
    coord_limits: float = 1.0,
):
    """Plot a single 3D pose.

    Args:
        pose (np.ndarray): a single pose of shape [n_keypoints, 2 or 3]
        skeleton (SkeletonProfile): class containing the skeleton information
    """
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=300)
        ax = fig.add_subplot(111, projection="3d")

    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    assert (
        pose.shape[0] == skeleton.n_keypoints
    ), f"Pose shape {pose.shape} does not match the skeleton {skeleton.n_keypoints}"
    assert pose.shape[1] == 3, f"Pose shape {pose.shape} should be [n_keypoints, 3]"

    n_keypoints = skeleton.n_keypoints
    kinematic_tree = skeleton.kinematic_tree_indices
    body_region_indices = skeleton.body_region_indices
    n_body_regions = len(skeleton.body_regions)
    n_chains = len(kinematic_tree)

    colormap = plt.get_cmap(colormap)
    if colors_chain is None:
        colors_chain = [colormap(i / n_chains) for i in range(n_chains)]
    elif isinstance(colors_chain, str):
        colors_chain = [colors_chain] * n_chains

    if colors_marker is None:
        colors_marker = [colormap(i / n_body_regions) for i in body_region_indices]

    assert (
        len(colors_chain) == n_chains and len(colors_marker) == n_keypoints
    ), "Color list should match the number of kinematic chains and keypoints"

    # Plot edges (segments)
    for chain, color in zip(kinematic_tree, colors_chain):
        ax.plot3D(
            *pose[chain].T,
            color=lineborder_color,
            zorder=2,
            alpha=alpha,
            linewidth=linewidth + lineborder,
        )
        ax.plot3D(
            *pose[chain].T, color=color, zorder=3, alpha=alpha, linewidth=linewidth,
        )
    # Plot nodes (keypoints)
    ax.scatter(
        *pose.T, c=colors_marker, s=marker_size, zorder=1, alpha=alpha, linewidth=0,
    )
    ax.scatter(
        *pose.T,
        c=colors_marker,
        s=marker_size + marker_edgewidth,
        zorder=4,
        alpha=alpha,
        edgecolor=marker_edgecolor,
        linewidth=0.5,
    )

    ax.set_xlim([-coord_limits, coord_limits])
    ax.set_ylim([-coord_limits, coord_limits])
    ax.set_zlim([-coord_limits, coord_limits])
    ax.set_axis_off()

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)


def visualize_pose(pose: np.ndarray, *args, **kwargs):
    if pose.shape[-1] == 2:
        visualize_pose2d(pose=pose, *args, **kwargs)
    elif pose.shape[-1] == 3:
        visualize_pose3d(pose=pose, *args, **kwargs)


def make_pose3d_seq_video(
    poseseq: np.ndarray,
    skeleton: SkeletonProfile,
    n_frames: int | None = None,
    fps: int = 50,
    savename: str = "pose3d.mp4",
    fig: matplotlib.figure.Figure = None,
    ax: matplotlib.axes.Axes = None,
    **kwargs,
):
    if fig is None or ax is None:
        fig = plt.figure(figsize=(80 * mm, 80 * mm), dpi=300)
        ax = fig.add_subplot(111, projection="3d")

    n_frames = len(poseseq) if n_frames is None else n_frames

    def animate(k):
        ax.clear()
        visualize_pose3d(pose=poseseq[k], skeleton=skeleton, ax=ax, **kwargs)
        ax.set_title(f"{skeleton.skeleton_name}: Frame {k}")

    anim = FuncAnimation(fig, animate, frames=np.arange(n_frames))

    ffmpeg_writer = animation.FFMpegWriter(fps=fps)
    anim.save(savename, writer=ffmpeg_writer)


def make_pose_seq_overlay(
    poseseq: np.ndarray,
    skeleton: SkeletonProfile,
    n_frames: int | None = None,
    n_samples: int = 10,
    alpha_min: float = 0.2,
    figsize: tuple = (80 * mm, 80 * mm),
    ax: matplotlib.axes.Axes = None,
    savename: str = "poseseq3d_overlay.png",
    **kwargs,
):
    datadim = poseseq.shape[-1]
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=300)
        if datadim == 3:
            ax = fig.add_subplot(111, projection="3d")
        elif datadim == 2:
            ax = fig.add_subplot(111)

    n_frames = len(poseseq) if n_frames is None else n_frames
    assert n_samples <= n_frames
    frames_to_plot = np.linspace(0, n_frames - 1, n_samples, dtype=int)

    for idx, frame in enumerate(frames_to_plot):
        visualize_pose(
            pose=poseseq[frame],
            skeleton=skeleton,
            ax=ax,
            alpha=alpha_min + (1 - alpha_min) * (idx / (n_samples - 1)),
            **kwargs,
        )
    if savename is not None:
        fig.savefig(savename, bbox_inches="tight")

    return ax


@hydra.main(config_path="../../configs", config_name="defaults.yaml", version_base=None)
def test_vis(cfg):
    from csbev.dataset.loader import prepare_datasets

    datasets = prepare_datasets(cfg)
    datasets = datasets["train"]

    for tag, dataset in datasets.items():
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        poseseq = dataset[0]["x"]

        plot_args = {
            "linewidth": 1.0,
            "marker_size": 20,
            "coord_limits": 0.8,
        }
        # plot_args = {
        #     "linewidth": 1.0,
        #     "marker_size": 20,
        #     "coord_limits": 0.8,
        #     "colors_chain": "gray",
        #     "lineborder_color": "gray",
        # }

        make_pose3d_seq_video(
            poseseq,
            dataset.skeleton,
            savename=f"/home/tianqingli/dl-projects/vqmap2/experiments/test/test_vis/poseseq_{tag}.mp4",
            **plot_args,
        )

        make_pose_seq_overlay(
            poseseq[:16],
            dataset.skeleton,
            n_samples=4,
            alpha_min=0.2,
            savename=f"/home/tianqingli/dl-projects/vqmap2/experiments/test/test_vis/poseseq_overlay_{tag}.png",
            **plot_args,
        )


if __name__ == "__main__":
    test_vis()
