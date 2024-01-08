import os
import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from loguru import logger
import tqdm


def visualize_latent_space(
    skeleton, to_plot, savepath, **kwargs
):
    num_codes = len(to_plot)
    for idx in tqdm.tqdm(range(num_codes)):
        visualize_seq(skeleton, to_plot[idx], idx, savepath, **kwargs)
    
    # load all images and merge
    full = []
    w = 8
    h = int(np.ceil(num_codes / w))
    overlap = 15
    for row in range(h):
        ims = [
            plt.imread(savepath+f'/code{idx}.png')[overlap:-overlap, overlap:-overlap, :]
            for idx in range(row*w, min((row+1)*w, num_codes))
        ]
        ims = np.concatenate(ims, axis=1)
        full.append(ims)
    full = np.concatenate(full, axis=0)

    plt.imsave(os.path.dirname(savepath)+f'/vis_codebook.png', full)
    plt.close()

def visualize_seq(
    skeleton, to_plot, idx, savepath,
    limits=10, stride=4, scale=3, node_size=30,
):
    mm = 1/25.4
    w, h = 60, 60
    fig = plt.figure(figsize=(w*mm, h*mm), dpi=200)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    n_joints = skeleton.num_keypoint
    kinematic_tree = skeleton.kinematic_tree

    colormap = plt.get_cmap(skeleton.colormap)
    colors = [colormap(e/len(kinematic_tree)) for e in range(len(kinematic_tree))]

    if n_joints in [18, 24]:
        ax.view_init(azim=180, elev=110)

    to_plot *= scale
    t = to_plot.shape[0]
    for frame in range(0, t, stride):
        alpha = 0.1 + 0.9*(frame/t)
        pose = to_plot[frame]
        joints_output = deepcopy(pose)
        joints_output = joints_output.reshape((-1, 3))
        
        for k in range(0, joints_output.shape[0]//n_joints):
            skeleton = joints_output[k*n_joints:(k+1)*n_joints]
            for chain, color in zip(kinematic_tree, colors):
                ax.plot3D(*skeleton[chain].T, color="k", zorder=2, alpha=alpha, linewidth=2.0+0.5)
                ax.plot3D(*skeleton[chain].T, color=color, zorder=3, alpha=alpha, linewidth=2.0)
            
            ax.scatter(
                *skeleton.T,
                c=np.arange(n_joints),
                cmap=colormap,
                s=node_size,
                zorder=1,
                alpha=0.1 + 0.9*(frame/t),
                linewidth=0,
            )
            ax.scatter(
                *skeleton.T,
                c=np.arange(n_joints),
                cmap=colormap,
                s=node_size+0.7,
                zorder=4,
                edgecolor="k",
                linewidth=0.5, alpha=0.1 + 0.9*(frame/t)
            )

        title = f" Code {idx}"
        ax.set_title(title)
        ax.set_xlim([-limits, limits])
        ax.set_ylim([-limits, limits])
        ax.set_zlim([-limits, limits])
        ax.set_axis_off()
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        fig.savefig(savepath+f'/code{idx}.png', transparent=True)
        plt.close()


def visualize(skeleton, to_plot, nframes, fname, titles):
    assert skeleton.num_keypoint == to_plot[0].shape[-2]
    ndim = to_plot[0].shape[-1]
    visualize_fcn = visualize3d if ndim == 3 else visualize2d
    return visualize_fcn(skeleton, to_plot, nframes, fname, titles)

def visualize3d(
    skeleton, to_plot, nframes, fname, titles,
    prior=0, limits=10, fps=50,
):
    axes = []
    n_cols = to_plot[0].shape[0] 
    n_rows = len(to_plot)
    
    fig = plt.figure(figsize=(4*n_cols, 4*n_rows), dpi=200)

    for i in range(n_rows*n_cols):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        axes.append(ax)

    n_joints = skeleton.num_keypoint
    kinematic_tree = skeleton.kinematic_tree
    colors = skeleton.colors
    colormap = plt.get_cmap(skeleton.colormap)
    if colors is None:
        colors = [colormap(e/len(kinematic_tree)) for e in range(len(kinematic_tree))]

    if n_joints in [18, 24]:
        for ax in axes:
            ax.view_init(azim=180, elev=110)
    
    def plot_pose(pose, ax, row):
        joints_output = deepcopy(pose)
        joints_output = joints_output.reshape((-1, 3))

        # ax.view_init(elev=0, azim=-90)
        ax.scatter(*joints_output.T, color='black')
        ax.scatter(*joints_output[0, :], color='red', s=50)
        
        for k in range(0, joints_output.shape[0]//n_joints):
            skeleton = joints_output[k*n_joints:(k+1)*n_joints]
            for chain, color in zip(kinematic_tree, colors):
                ax.plot3D(*skeleton[chain].T, color="k", zorder=2)
                ax.plot3D(*skeleton[chain].T, color=color, zorder=3)

        if row == 0:
            title_color = 'green'
            if i < prior:
                title = 'GT'
                title_color = 'blue'
            else:
                title = titles[row]
            title += f" Frame {i}"
            ax.set_title(title, c=title_color)
        ax.grid(False)

        # make the panes transparent, keep ground
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        ax.set_xlim([-limits, limits])
        ax.set_ylim([-limits, limits])
        ax.set_zlim([0, limits*2])
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    
    def animate(i):
        for col in range(n_cols):
            for row in range(n_rows):
                pose = to_plot[row][col]
                ax = axes[row*n_cols+col]        
                ax.clear()
                plot_pose(pose[i], ax, row)
                
        plt.tight_layout()

    # create animation using the animate() function
    anim = FuncAnimation(fig, animate, frames=np.arange(nframes))

    ffmpeg_writer = animation.FFMpegWriter(fps=fps)
    logger.info(f"Visualization video saved to {fname}")
    anim.save(fname, writer=ffmpeg_writer)
    
    plt.close(fig)
    
    return anim

def visualize2d(skeleton, to_plot, nframes, fname, titles):
    axes = []
    n_cols = to_plot[0].shape[0] 
    n_rows = len(to_plot)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=200)
    try:
        axes = axes.flatten()
    except:
        axes = [axes]
    
    n_joints = to_plot[0][0].shape[1]
    if n_joints == 7:
        # calms21 
        # kinematic_tree = [
        #     [0, 1, 3, 2, 0],
        #     [0, 4, 6, 5, 0],
        # ]
        kinematic_tree = [
            [0, 3], [3, 1], [3, 2],
            [0, 6], [0, 4], [0, 5]
        ]
        colors = ['red', 'green', 'green', 'blue']
        fps = 30
    elif n_joints == 8:
        kinematic_tree = [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
            [4, 5], [6, 4], [7, 4]
        ]

    keypoint_colormap = 'RdYlGn'
    cmap = plt.get_cmap(keypoint_colormap)
    fps = 30
    node_size = 30
    
    def animate(i):
        for col in range(n_cols):
            for row in range(n_rows):
            
                pose = to_plot[row][col]
            
                ax = axes[row*n_cols+col]        
                ax.clear()
                
                joints_output = deepcopy(pose[i])

                ax.scatter(joints_output[:, 0], joints_output[:, 1], color='black')
                
                for k in range(0, joints_output.shape[0]//n_joints):

                    skeleton = joints_output[k*n_joints:(k+1)*n_joints]
                    for e, chain in enumerate(kinematic_tree):
                        ax.plot(
                            *skeleton[chain].T,
                            color="k",
                            zorder=2,
                        )
                        ax.plot(
                            *skeleton[chain].T,
                            color=cmap(e/n_joints),
                            zorder=3,
                        )

                ax.scatter(
                    *joints_output.T,
                    c=np.arange(n_joints),
                    cmap=cmap,
                    s=node_size,
                    zorder=1,
                    linewidth=0,
                )
                ax.scatter(
                    *joints_output.T,
                    c=np.arange(n_joints),
                    cmap=cmap,
                    s=node_size,
                    zorder=4,
                    edgecolor="k",
                )

                title = f" Frame {i}"
                ax.set_title(title)
                ax.axis('equal')
                ax.set_axis_off()
                ax.grid(False)
                
                if row == 0:
                    title_color = 'green'
                    title = titles[row]
                    title += f" Frame {i}"
                    ax.set_title(title, c=title_color)
                
                plt.tight_layout()

    # create animation using the animate() function
    anim = FuncAnimation(fig, animate, frames=np.arange(nframes))

    ffmpeg_writer = animation.FFMpegWriter(fps=fps)
    print(f"Visualization saved to {fname}")
    anim.save(fname, writer=ffmpeg_writer)
    
    # plt.close(fig)
    
    return anim 