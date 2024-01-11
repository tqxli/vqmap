from vqmap.utils.quaternion import *
import scipy.ndimage.filters as filters
import os
from loguru import logger
from vqmap.config.config import parse_config


def skeleton_initialize(
    skeleton_name='rat23',
    skeleton_root='./data/skeletons'
):
    assert os.path.exists(skeleton_root), f"{skeleton_root} does not exist"

    _profiles = [f.split('.yaml')[0] for f in os.listdir(skeleton_root) if f.endswith('.yaml')]
    assert skeleton_name in _profiles, f"{skeleton_name} does not exist in {skeleton_root}"
    
    info = parse_config(os.path.join(skeleton_root, f"{skeleton_name}.yaml"))
    
    profile = PoseProfile(skeleton_name, info)
    
    return profile
    

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def rotation_matrix_from_vectors(vec1, vec2):
    ndim = vec1.shape[-1]
    if ndim == 2:
        return _rotate_vec_2d(vec1, vec2)
    elif ndim == 3:
        return _rotate_vec_3d(vec1, vec2)
    else:
        raise Exception


def _rotate_vec_3d(vec1, vec2):
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


def _rotate_vec_2d(vec1, vec2):
    """ Same as above but 2D
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(2), (vec2 / np.linalg.norm(vec2)).reshape(2)
    x1, y1 = a
    x2, y2 = b
    c = np.dot(a, b)
    rotation_matrix = [[c, x2*y1-x1*y2], [x1*y2-x2*y1, c]]
    return rotation_matrix


class PoseProfile:
    def __init__(self, name, info):
        self.name = name
        self.info = info
        
        joint_names = info["keypoint_names"]
        self.num_keypoint = info["num_keypoint"]

        self.kinematic_tree = info["kinematic_tree"]

        self.anterior, self.posterior = info["anterior"], info["posterior"]
        self.indices_reorder = indices_reorder = np.array(info["indices_reorder"])
        self.joint_names = [joint_names[idx] for idx in indices_reorder]
        
        self.offsets = info.get("offsets", None)
        logger.info(f"Pose profile: {name}")
        logger.info(f"Number of keypoints: {self.num_keypoint}")
        logger.info("Kinematic tree: ")
        for chain in self.kinematic_tree:
            logger.info(f'     Chain: {[self.joint_names[idx] for idx in chain]}')
        logger.info(f"Align direction to +x: {self.joint_names[self.anterior]}-{self.joint_names[self.posterior]}")

        # visualization
        self.colors = info.get("colors", None)
        self.colormap = info.get("colormap", "RdYlGn")

    def align_pose(self, poses, center_only=False, align_z=False):
        ndim = poses.shape[-1]
        poses = poses[:, self.indices_reorder, :]
        traj = poses[:, :1]
        poses = poses - traj
        
        if center_only:
            return poses, traj
        
        spineline = poses[:, self.anterior] - poses[:, self.posterior]
        spineline = spineline[:, None, :]
        if not align_z and ndim == 3:
            spineline[:, :, 2] = 0
        spineline = unit_vector(spineline)

        # by default, align heading to the +x axis
        x_axis = np.zeros_like(spineline)
        x_axis[:, :, 0] = 1
    
        # rotation matrices
        rotmat = [rotation_matrix_from_vectors(vec1, vec2) for (vec1, vec2) in zip(spineline, x_axis)]
        rotmat = np.stack(rotmat, 0)

        poses_rot = rotmat @ poses.transpose((0, 2, 1))
        poses_rot = poses_rot.transpose((0, 2, 1))
        
        return poses_rot, traj

    def inverse_kinematics_np(self, joints, smooth_forward=False):
        '''Get Forward Direction'''
        forward = joints[:, self.anterior] - joints[:, self.posterior] 

        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[1,0,0]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        for chain in self.kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self.offsets[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
                # print(u.shape)
                # (batch, 3)
                v = joints[:, chain[j+1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                # print(u.shape, v.shape)
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:,chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params
    
    def convert_to_euclidean(self, inputs):
        n_samples, seqlen, _ = inputs.shape
        inputs = inputs.reshape(n_samples*seqlen, self.num_keypoint, -1)
        n_chan = inputs.shape[-1]

        # quaternion
        if n_chan == 4:
            inputs = self.forward_kinematics_np(
                inputs, 
                np.zeros((inputs.shape[0], 3)), do_root_R=True
            )
        # continuous 6D rotation
        elif n_chan == 6:
            inputs = self.forward_kinematics_cont6d_np(
                inputs, 
                np.zeros((inputs.shape[0], 3)), do_root_R=True
            )
        # otherwise keep xyz
        inputs = inputs.reshape((n_samples, seqlen, self.num_keypoint, -1))
        return inputs
    

if __name__ == "__main__":
    profile = skeleton_initialize()
    
    from vqmap.datasets.base import MocapContBase
    
    datapath = [
        '/media/mynewdrive/datasets/dannce/social_rat/SCN2A_WK1/2022_09_15_M1/SDANNCE/bsl0.5_FM/save_data_AVG0.mat'
    ]
    seqlen = 50
    
    dataset = MocapContBase(datapath, seqlen)
    sample = dataset[0]
    print(sample[0].shape)