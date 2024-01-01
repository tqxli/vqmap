import numpy as np
import torch

REORDER_INDICES = {
    23: [4, 3, 1, 2, 0, *np.arange(5, 23)],
    22: [4, 3, 0, 1, 2, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 17, 16, 21, 20, 19],
}
SPINELINES = {
    23: [1, 0],
    22: [0, 5]
}

# Spine M - Spine F should always be + x direction
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def align_pose(sample, first_frame_only=False, remove_zoffsets=True):
    n_joints = sample.shape[1]
    sample = sample[:, REORDER_INDICES[n_joints], :]
    traj = sample[:, :1]
    sample = sample - traj
    spineline_indices = SPINELINES[n_joints]
    spineline = sample[:, spineline_indices[0]:spineline_indices[0]+1] - sample[:, spineline_indices[1]:spineline_indices[1]+1]
    if remove_zoffsets:
        spineline[:, :, 2] = 0
    spineline = unit_vector(spineline)

    x_axis = np.zeros_like(spineline)
    x_axis[:, :, 0] = 1
 
    rotmat = [rotation_matrix_from_vectors(vec1, vec2) for (vec1, vec2) in zip(spineline, x_axis)]
    rotmat = np.stack(rotmat, 0)
    
    if first_frame_only:
        rotmat = rotmat[:1, :, :]

    rotmat = torch.from_numpy(rotmat).float()
    sample_rot = rotmat @ sample.permute(0, 2, 1)
    sample_rot = sample_rot.permute(0, 2, 1)
    
    sample_rot /= 10.0
    
    return sample_rot, rotmat, traj