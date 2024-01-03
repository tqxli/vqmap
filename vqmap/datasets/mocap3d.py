from vqmap.datasets.base import *


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
    cfg = OmegaConf.load('configs/dataset_chunk.yaml')
    cfg.dataset.root = '/home/tianqingli/dl-projects/ACTOR/data/uestc/data_frontview.npy'
    datapath = cfg.dataset.root
    dataset = UESTC(datapath, cfg.dataset)
    poseseq, _ = dataset[0]
    print(poseseq.shape)
    print(f"Total samples: {len(dataset)}")