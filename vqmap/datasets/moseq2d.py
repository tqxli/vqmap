import os
import numpy as np
import pandas as pd
import glob
import tqdm
import torch
from torch.utils.data import Dataset
from vqmap.datasets.utils import unit_vector
from vqmap.datasets.calms import rotation_matrix_from_vectors

kinematic_tree = [
    [0, 1, 2, 3, 4, 5],
    [6, 4, 5],
    [7, 4, 5]
]
center = 2
alignment = [0, 5]

def align_pose2d(sample):
    traj = sample[:, center:center+1]
    sample = sample - traj
    spineline = sample[:, alignment[1]:alignment[1]+1] - sample[:, alignment[0]:alignment[0]+1]
    spineline = unit_vector(spineline)

    x_axis = np.zeros_like(spineline)
    x_axis[:, :, 0] = 1
 
    rotmat = [rotation_matrix_from_vectors(vec1, vec2) for (vec1, vec2) in zip(spineline, x_axis)]
    rotmat = np.stack(rotmat, 0)

    sample_rot = rotmat @ sample.transpose((0, 2, 1))
    sample_rot = sample_rot.transpose((0, 2, 1))
    
    sample_rot /= 10.0
    
    return sample_rot, rotmat, traj

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
    
class Moseq2D(Dataset):
    def __init__(self,
                 datapath='/media/mynewdrive/datasets/keypoint-moseq/dlc_project/videos',
                 seqlen=64,
                 kind='xyz',
                 stride=1,
                 body_splits=None
    ):
        super().__init__()
        
        self.datapath = datapath
        self.seqlen = seqlen
        self.max_len = seqlen

        self.stride = stride
        
        self.body_splits = body_splits
        self.do_body_splits = body_splits is not None
        
        self._load_data()
    
    def _load_data(self):
        filepaths = list_files_with_exts(
            self.datapath,
            [".csv", ".h5", ".hdf5"]
        )
 
        coordinates, confidences = {}, {}
        for filepath in tqdm.tqdm(filepaths, desc=f"Loading keypoints", ncols=72):
            name = _name_from_path(
                filepath, False, '-', True
            )
            new_coordinates, new_confidences, bodyparts = _deeplabcut_loader(
                filepath, name
            )
            
            coordinates.update(new_coordinates)
            confidences.update(new_confidences)
        
        # preprocess
        for k, coords in tqdm.tqdm(coordinates.items(), desc=f"Aligning", ncols=72):
            maxlen = self.seqlen * (coords.shape[0] // self.seqlen)
            coords = coords[:maxlen, 1:]
            coords = align_pose2d(coords)[0]
            coordinates[k] = coords
        
        self.joints = torch.from_numpy(np.concatenate(list(coordinates.values()), 0))
        print(self.joints.shape)

    def __len__(self):
        return self.joints.shape[0] // self.seqlen
    
    def __getitem__(self, index):
        motion = self.joints[index*self.stride:index*self.stride+self.seqlen]
        motion = motion.reshape((motion.shape[0], -1))
        
        if self.do_body_splits:
            return [motion[bs] for bs in self.body_splits], None
        
        return motion, None
    

if __name__ == "__main__":
    dataset = Moseq2D()
    print(len(dataset), dataset[0][0].shape)
    
    