import pandas as pd
import glob
from tqdm import tqdm
from vqmap.datasets.base import *


class MocapCont2D(MocapContBase):
    def _load_data(self):
        """ Load raw motion capture data from the List datapath
        """
        self.pose3d, self.num_frames = [], []
        self.raw_num_frames = 0
        data_all = self._load(self.datapath)

        for data in data_all:
            data = data[::self.downsample] * self.scale
            data = self._trim(data)
            data = self._align(data)
            data = self._normalize(data)
            self.pose3d.append(data)
        
        self.pose3d = torch.from_numpy(np.concatenate(self.pose3d)).flatten(1)
        self.pose_dim = self.pose3d.shape[-1]
        logger.info(f"Dataset chunking: {self.raw_num_frames} --> {self.pose3d.shape}")
    
    def _load(self, datapath):
        data = np.load(datapath, allow_pickle=True)[()]
        joints = [
            v['keypoints'].transpose((1, 0, 3, 2))
            for v in data['annotator-id_0'].values()] #([n_frames, 2-mice, 2, 7)]

        joints = [[js[0], js[1]] for js in joints]
        return sum(joints, [])


def list_files_with_exts(filepath_pattern, ext_list, recursive=True):
    if isinstance(filepath_pattern, list):
        matches = []
        for fp in filepath_pattern:
            matches += list_files_with_exts(fp, ext_list, recursive=recursive)
        return sorted(set(matches))

    else:
        # make sure extensions all start with "." and are lowercase
        ext_list = ["." + ext.strip(".").lower() for ext in ext_list]

        if os.path.isdir(filepath_pattern):
            filepath_pattern = os.path.join(filepath_pattern, "*")

        # find all matches (recursively)
        matches = glob.glob(filepath_pattern)
        if recursive:
            for match in list(matches):
                matches += glob.glob(os.path.join(match, "**"), recursive=True)

        # filter matches by extension
        matches = [
            match
            for match in matches
            if os.path.splitext(match)[1].lower() in ext_list
        ]
        return matches

def _deeplabcut_loader(filepath, name):
    """Load tracking results from deeplabcut csv or hdf5 files."""
    ext = os.path.splitext(filepath)[1]
    if ext == ".h5":
        df = pd.read_hdf(filepath)
    if ext == ".csv":
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)

    coordinates, confidences = {}, {}
    bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
    if "individuals" in df.columns.names:
        for ind in df.columns.get_level_values("individuals").unique():
            ind_df = df.xs(ind, axis=1, level="individuals")
            arr = ind_df.to_numpy().reshape(len(ind_df), -1, 3)
            coordinates[f"{name}_{ind}"] = arr[:, :, :-1]
            confidences[f"{name}_{ind}"] = arr[:, :, -1]
    else:
        arr = df.to_numpy().reshape(len(df), -1, 3)
        coordinates[name] = arr[:, :, :-1]
        confidences[name] = arr[:, :, -1]

    return coordinates, confidences, bodyparts

def _name_from_path(filepath, path_in_name, path_sep, remove_extension):
    """Create a name from a filepath.

    Either return the name of the file (with the extension removed) or return
    the full filepath, where the path separators are replaced with `path_sep`.
    """
    if remove_extension:
        filepath = os.path.splitext(filepath)[0]
    if path_in_name:
        return filepath.replace(os.path.sep, path_sep)
    else:
        return os.path.basename(filepath)

class MoSeqDLC2D(MocapCont2D):
    def _load(self, datapath):
        filepaths = list_files_with_exts(
            datapath,
            [".csv", ".h5", ".hdf5"]
        )
 
        coordinates, confidences = {}, {}
        for filepath in tqdm(filepaths, desc=f"Loading keypoints", ncols=72):
            name = _name_from_path(
                filepath, False, '-', True
            )
            new_coordinates, new_confidences, bodyparts = _deeplabcut_loader(
                filepath, name
            )
            coordinates.update(new_coordinates)
            confidences.update(new_confidences)
        
        joints = [val[:, 1:] for val in coordinates.values()]
        return joints

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import os
    cfg = OmegaConf.load('configs/dataset.yaml')
    cfg.dataset.root = '/media/mynewdrive/datasets/CalMS21/mab-e-baselines-master/data/calms21_task1_train.npy'
    cfg.dataset.root = root = '/media/mynewdrive/datasets/keypoint-moseq/dlc_project/videos'
    datapath = root
    cfg.dataset.skeleton = 'moseq_dlc2d'
    dataset = MoSeqDLC2D(datapath, cfg.dataset)
    poseseq, _ = dataset[0]
    print(poseseq.shape)
    print(f"Total samples: {len(dataset)}")