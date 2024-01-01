import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from vqmap.utils.skeleton import *


def visualize(to_plot, nframes, fname, titles, prior=0):
    axes = []
    n_cols = to_plot[0].shape[0] 
    n_rows = len(to_plot)
    
    fig = plt.figure(figsize=(4*n_cols, 4*n_rows), dpi=200)

    for i in range(n_rows*n_cols):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        axes.append(ax)

    n_joints = to_plot[0].shape[-2]
    if n_joints == 23:
        kinematic_tree = KINEMATIC_TREE
        colors = ['orange', 'orange', 'green', 'green', 'blue', 'blue']
        limits = 10
        fps = 50
    elif n_joints == 20:
        kinematic_tree = KINEMATIC_TREE_MOUSE22
        colors = ['orange', 'orange', 'green', 'green', 'blue', 'blue']
        limits = 10
        fps = 50
    elif n_joints == 24:
        kinematic_tree = KINEMATIC_TREE_HUMAN
        colors = ['red', 'red', 'green', 'blue', 'blue']
        limits = 1
        fps = 30
    elif n_joints == 18:
        kinematic_tree = KINEMATIC_TREE_UESTC
        colors = ['red', 'red', 'green', 'blue', 'blue']
        limits = 10
        fps = 30
    elif n_joints == 13:
        kinematic_tree = KINEMATIC_TREE_OMS
        colors = ['red', 'orange', 'green', 'blue', 'blue', 'blue']
        limits = 1.2
        fps = 30
    else:
        kinematic_tree = KINEMATIC_TREE_RAT7M
        colors = ['orange', 'orange', 'green', 'green', 'blue', 'blue']
        limits = 10
        fps = 30
    
    if n_joints in [18, 24]:
        for ax in axes:
            ax.view_init(azim=180, elev=110)
    
    def animate(i):
        for col in range(n_cols):
            for row in range(n_rows):
            
                pose = to_plot[row][col]
            
                ax = axes[row*n_cols+col]        
                ax.clear()
                
                joints_output = deepcopy(pose[i])
                joints_output = joints_output.reshape((-1, 3))

                # ax.view_init(elev=0, azim=-90)
                ax.scatter(joints_output[:, 0], joints_output[:, 1], joints_output[:, 2], color='black')
                ax.scatter(joints_output[0, 0], joints_output[0, 1], joints_output[0, 2], color='red', s=50)
                
                for k in range(0, joints_output.shape[0]//n_joints):
                    skeleton = joints_output[k*n_joints:(k+1)*n_joints]
                    for chain, color in zip(kinematic_tree, colors):
                        ax.plot3D(
                            skeleton[chain,0],
                            skeleton[chain,1], 
                            skeleton[chain,2], linewidth=2.0, color=color)

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
                # ax.set_axis_off()
                # make the panes transparent
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
                
                plt.tight_layout()

    # create animation using the animate() function
    anim = FuncAnimation(fig, animate, frames=np.arange(nframes))

    ffmpeg_writer = animation.FFMpegWriter(fps=fps)
    print(f"Visualization saved to {fname}")
    anim.save(fname, writer=ffmpeg_writer)
    
    plt.close(fig)
    
    return anim

def visualize2d(to_plot, nframes, fname, titles):
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