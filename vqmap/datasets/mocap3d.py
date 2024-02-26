from vqmap.datasets.base import *


class MocapSimpleCompiled(MocapContBase):
    def _load_data(self):
        predictions = np.load(self.datapath, allow_pickle=True)[()]
        self.pose3d, self.num_frames = [], []
        self.raw_num_frames = 0
        
        for expname, data in tqdm(predictions.items(), desc='Preprocess'):
            self.raw_num_frames += len(data)
            data = data[::self.downsample] * self.scale
            data = self._trim(data)
            data = self._align(data)
            data = self._normalize(data)
            data = self._convert(self.data_rep, data)
            
            self.pose3d.append(data)
                
        self.pose3d = torch.from_numpy(np.concatenate(self.pose3d)).flatten(1)
        self.pose_dim = self.pose3d.shape[-1]
        logger.info(f"Dataset chunking: {self.raw_num_frames} --> {self.pose3d.shape}")
        
        self.cfg.num_frames = self.num_frames


class PupDevelopment(MocapContBase):
    def _load_data(self):
        predictions = np.load(self.datapath, allow_pickle=True)[()]
        self.pose3d, self.num_frames = [], []
        self.raw_num_frames = 0
        
        indices = np.array(list(np.arange(5)) + list(np.arange(8, 16)))

        for day in tqdm(predictions, desc=f'Preprocess'):
            for expname, data in predictions[day].items():
                self.raw_num_frames += len(data)
                data = data[::self.downsample] * self.scale
                data = data[:, indices]
                data = self._trim(data)
                data = self._align(data)
                data = self._normalize(data)
                data = self._convert(self.data_rep, data)
                
                self.pose3d.append(data)
                
        self.pose3d = torch.from_numpy(np.concatenate(self.pose3d)).flatten(1)
        self.pose_dim = self.pose3d.shape[-1]
        logger.info(f"Dataset chunking: {self.raw_num_frames} --> {self.pose3d.shape}")
        
        self.cfg.num_frames = self.num_frames    
        self.cfg.expnames = [[exp for exp in predictions[day]] for day in predictions]


class UESTC(MocapChunkBase):
    def _load_data(self):
        self.joints_idx = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]
        data = np.load(self.datapath, allow_pickle=True)[()]

        pose3d = data['joints3d']
        self.num_joints = pose3d[0].shape[1]
        self.pose3d, self._num_frames = [], []
        for poseseq in pose3d:
            poseseq = np.stack([-poseseq[:, :, 1], poseseq[:, :, 0], poseseq[:, :, 2]], axis=-1)
            poseseq = poseseq[:, self.joints_idx]
            poseseq -= poseseq[:, :1]
            poseseq = poseseq.reshape((poseseq.shape[0], -1)) * 8

            self.pose3d.append(poseseq)
            self._num_frames.append(poseseq.shape[0])
            
        self.actions = np.zeros((len(self.pose3d)))
        assert len(self.pose3d) == len(self.actions)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/dataset/pupdev.yaml')
    # cfg.dataset.root = '/home/tianqingli/dl-projects/ACTOR/data/uestc/data_frontview.npy'
    datapath = cfg.dataset.root
    dataset = PupDevelopment(datapath, cfg.dataset)
    poseseq, _ = dataset[0]
    print(poseseq.shape)
    print(f"Total samples: {len(dataset)}")
    
    from vqmap.utils.visualize import visualize, visualize_seq
    from vqmap.utils.skeleton import skeleton_initialize
    poseseq = poseseq.reshape((poseseq.shape[0], -1, 3)).numpy()
    skeleton = skeleton_initialize('pupdev_notail')

    def plot_pose(joints_output, ax, limits=1):
        joints_output = joints_output.reshape((-1, 3))
        ax.scatter(*joints_output.T, color='black')

        for chain, color in zip(skeleton.kinematic_tree, skeleton.colors):
            ax.plot3D(*joints_output[chain].T, color="k", zorder=2)
            ax.plot3D(*joints_output[chain].T, color=color, zorder=3)
        
        ax.grid(False)

        # make the panes transparent, keep ground
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        ax.set_xlim([-limits, limits])
        ax.set_ylim([-limits, limits])
        ax.set_zlim([-limits, limits])
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg') 
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_pose(poseseq[50], ax, limits=5)
    plt.show()