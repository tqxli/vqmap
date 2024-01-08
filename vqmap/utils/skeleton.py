from vqmap.utils.quaternion import *
import scipy.ndimage.filters as filters
import os
from loguru import logger
from vqmap.config.config import parse_config

KINEMATIC_TREE = [
    [0, 1, 2, 3, 4], #[SpineM, SpineF, EarL, EarR, Snout]
    [0, 5, 6], #[SpineM, SpineF, TailBase]
    [1, 7, 8, 9, 10], # [SpineF, ShoulderL, ElbowL, WristL, HandL]
    [1, 11, 12, 13, 14], # right
    [5, 15, 16, 17, 18], # [SpineL, HipL, KneeL, AnkleL, FootL]
    [5, 19, 20, 21, 22] # right
]
OFFSETS = np.array([
    [0,0,0], #spineM
    [1,0,0], # spineF
    [1,0,0], # earL
    [1,0,0], # earR
    [1,0,0], # snout
    [-1,0,0], # spineL
    [-1,0,0], # tail
    [0,1,0], # left arm
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,-1,0], # right arm
    [0,-1,0],
    [0,-1,0],
    [0,-1,0],
    [0,1,0], # left leg
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,-1,0], # right leg
    [0,-1,0],
    [0,-1,0],
    [0,-1,0],]
)
KINEMATIC_TREE_RAT7M = [
    # [2, 3, 4, 1],
    # [1, 0, 5],
    # [0, 6, 7],
    # [5, 8, 17, 18], [5, 9, 16, 19],
    # [1, 12, 10, 11], [1, 13, 14, 15],
    [0, 1, 4, 3, 2],
    [0, 5],
    [1, 6, 7, 8], [1, 9, 10, 11],
    [5, 12, 13, 14], [5, 15, 16, 17]
]
OFFSETS_RAT7M = np.array([
    [0,0,0], #medial spine
    [1,0,0], # anterior spine
    [1,0,0], # front head
    [1,0,0], #back of head
    [1,0,0], # left of head
    [-1,0,0], # posterior spine
    [0,0,1], #offset1
    [0,0,1], # offset2
    [0,1,0], #left hip
    [0,-1,0], # right hip
    [0,1,0], #left elbow
    [0,1,0], # left arm
    [0,1,0], # left shoulder
    [0,-1,0], # right shoulder
    [0,-1,0], #right elbow
    [0,-1,0], # right arm
    [0,-1,0], #right knee
    [0,1,0], # left knee
    [0,1,0], #left shin
    [0,-1,0], # right shin
])


KINEMATIC_TREE_MOUSE22 = [
    [0, 1, 2, 3, 4],
    [0, 5],
    [1, 6, 7, 8, 9],
    [1, 10, 11, 12, 13],
    [0, 14, 15, 16],
    [0, 17, 18, 19]
]

KINEMATIC_TREE_HUMAN = [
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20, 22],
    [9, 14, 17, 19, 21, 23]  # same as smpl
]


KINEMATIC_TREE_UESTC = [[0, 12, 13, 14, 15],
                        [0, 9, 10, 11, 16],
                        [0, 1, 8, 17],
                        [1, 5, 6, 7],
                        [1, 2, 3, 4]]

KINEMATIC_TREE_OMS = [
    [0, 1, 2],
    [0, 3, 4], 
    [0, 5, 6], [0, 7, 8],
    [3, 9, 10], [3, 11, 12],
]

class Skeleton(object):
    def __init__(self, offset, kinematic_tree, device):
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def njoints(self):
        return len(self._raw_offset)

    def offset(self):
        return self._offset

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._parents

    # joints (batch_size, joints_num, 3)
    def get_offsets_joints_batch(self, joints):
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[:, None] * _offsets[:, i]

        self._offset = _offsets.detach()
        return _offsets

    # joints (joints_num, 3)
    def get_offsets_joints(self, joints):
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            # print(joints.shape)
            _offsets[i] = torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0) * _offsets[i]

        self._offset = _offsets.detach()
        return _offsets

    def inverse_kinematics_animal_np(self, joints, smooth_forward=False):
        '''Get Forward Direction'''
        spineM, spineL = 0, 1
        forward = joints[:, spineM] - joints[:, spineL] 

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
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
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

    # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
    # joints (batch_size, joints_num, 3)
    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        assert len(face_joint_idx) == 4
        '''Get Forward Direction'''
        l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        # forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward = np.cross(np.array([[0, 0, 1]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        '''Get Root Rotation'''
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        '''Inverse Kinematics'''
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_offset_np[chain[j+1]][np.newaxis,...].repeat(len(joints), axis=0)
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

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(len(quat_params), -1).detach().to(self.device)
            for i in range(1, len(chain)):
                R = qmul(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i-1]]
        return joints

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics_np(self, quat_params, root_pos, skel_joints=None, do_root_R=True):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints

    def forward_kinematics_cont6d(self, cont6d_params, root_pos, skel_joints=None, do_root_R=True):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            # skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = torch.eye(3).expand((len(cont6d_params), -1, -1)).detach().to(cont6d_params.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i-1]]
        return joints


def skeleton_initialize(kind='rat23'):
    if kind == 'rat23':
        offsets = OFFSETS 
        kinematic_tree = KINEMATIC_TREE
    else:
        offsets = OFFSETS_RAT7M
        kinematic_tree = KINEMATIC_TREE_RAT7M
    animal_skeleton = Skeleton(
        offset=torch.from_numpy(offsets),
        kinematic_tree=kinematic_tree,
        device='cpu'
    )
    animal_skeleton.set_offset(torch.from_numpy(offsets))
    
    return animal_skeleton

def get_njoints(config_dataset):
    if config_dataset.name == 'rat7m':
        n_joints = 20
    elif config_dataset.name == 'calms21':
        n_joints = 7
    elif config_dataset.name == 'moseq2d':
        n_joints = 8
    elif config_dataset.name == "mocap_mouse22":
        n_joints = 20
    elif config_dataset.name == "humanact12":
        n_joints = 24
    elif config_dataset.name == "uestc":
        n_joints = 18
    elif config_dataset.name == "oms":
        n_joints = 13
    else:
        n_joints = 23
    
    return n_joints


def skeleton_initialize_v2(
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

    def align_pose(self, poses, align_z=False):
        ndim = poses.shape[-1]
        poses = poses[:, self.indices_reorder, :]
        traj = poses[:, :1]
        poses = poses - traj
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
    profile = skeleton_initialize_v2()
    
    from vqmap.datasets.base import MocapContBase
    
    datapath = [
        '/media/mynewdrive/datasets/dannce/social_rat/SCN2A_WK1/2022_09_15_M1/SDANNCE/bsl0.5_FM/save_data_AVG0.mat'
    ]
    seqlen = 50
    
    dataset = MocapContBase(datapath, seqlen)
    sample = dataset[0]
    print(sample[0].shape)