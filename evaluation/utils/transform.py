import copy
import numpy as np
from pyquaternion import Quaternion
from typing import List, Union
from math import cos, sin
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

def EulerAngles2RotationMatrix(EulerAngles: Union[List[float], np.array]) -> np.ndarray:
    """
    Convert Euler angles to RotationMatrix.
    """
    theta = EulerAngles
    R_x = np.array([[1,         0,                  0               ],
                    [0,         cos(theta[0]),      -sin(theta[0])  ],
                    [0,         sin(theta[0]),      cos(theta[0])   ]
                    ])

    R_y = np.array([[cos(theta[1]),     0,      sin(theta[1])   ],
                    [0,                 1,      0               ],
                    [-sin(theta[1]),    0,      cos(theta[1])   ]
                    ])
                
    R_z = np.array([[cos(theta[2]),     -sin(theta[2]),     0   ],
                    [sin(theta[2]),     cos(theta[2]),      0   ],
                    [0,                 0,                  1   ]
                    ])
                    
    RotationMatrix = np.dot(R_z, np.dot(R_y, R_x))
    return RotationMatrix

def EulerAnglesXYZ2TransformationMatrix(EulerAngles: Union[List[float], np.array], XYZ: Union[List[float], np.array]) -> np.ndarray:
    """
    Give EulerAngles and XYZ to get a homogeneous transformation matrix.

    Arguements:
        EulerAngles {Union[List[float], np.array]} -- EulerAngles
        XYZ {Union[List[float], np.array]} -- XYZ value
    
    Returns:
        np.array -- Homogeneous transformation matrix
    """
    TransformationMatrix = np.zeros((4, 4))
    RotationMatrix = EulerAngles2RotationMatrix(EulerAngles)
    TransformationMatrix[:3, :3] = RotationMatrix
    TransformationMatrix[:3, 3] = XYZ
    TransformationMatrix[3, 3] = 1
    return TransformationMatrix

def TransformationMatrix2QuaternionXYZ(TransformationMatrix: Union[List[float], np.array]) -> np.ndarray:
    """
    Give a homogeneous transformation matrix to get Quaternion and XYZ.
    """
    T = np.asarray(TransformationMatrix)
    R = T[:3,:3]
    q = Quaternion(matrix=R)
    xyz = T[:-1,3]
    return (
        np.asarray([
            q[0], q[1], q[2], q[3]
        ]),
        xyz
    )

def QuaternionXYZ2TransformationMatrix(Quaternion: Union[List[float], np.array], XYZ: Union[List[float], np.array]) -> np.ndarray:
    """
    Give Quaternion and XYZ to get a homogeneous transformation matrix.

    :param Quaternion Union[List[float], np.array]: Quaternion [w, x, y, z]
    :param XYZ Union[List[float], np.array]: XYZ value
    :rtype np.array: Homogeneous transformation matrix
    """
    q = [0, 0, 0, 0]
    q[0] = Quaternion[1]
    q[1] = Quaternion[2]
    q[2] = Quaternion[3]
    q[3] = Quaternion[0]
    TransformationMatrix = np.zeros((4, 4))
    RotationMatrix = np.asarray(R.from_quat(q).as_matrix())
    TransformationMatrix[:3, :3] = RotationMatrix
    TransformationMatrix[:3, 3] = XYZ
    TransformationMatrix[3, 3] = 1
    return TransformationMatrix

def transform_pointcloud_numpy(
    points: np.ndarray, # [N, x, y, z]
    transformation_matrix: np.ndarray, # 4 x 4
) -> np.ndarray:
    """
    Matrix transformation of points.
    """
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.transform(transformation_matrix)
    return np.asarray(pointcloud.points)

def transform_pointcloud_torch(pointcloud, transformation_matrix, in_place=True):
    """
    Parameters
    ----------
    pc: A pytorch tensor pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        M could be some additional mask dimensions or whatever, but the
        3 are x-y-z
    transformation_matrix: A 4x4 homography

    Returns
    -------
    Mutates the pointcloud in place and transforms x, y, z according the homography
    """
    pc = pointcloud
    assert isinstance(pc, torch.Tensor)
    assert type(pc) == type(transformation_matrix)
    assert pc.ndim == transformation_matrix.ndim
    if pc.ndim == 3:
        N, M = 1, 2
    elif pc.ndim == 2:
        N, M = 0, 1
    else:
        raise Exception("Pointcloud must have dimension Nx3 or BxNx3")
    xyz = pc[..., :3]
    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    homogeneous_xyz = torch.cat((xyz, torch.ones(ones_dim, device=xyz.device)), dim=M)
    transformed_xyz = torch.matmul(
        transformation_matrix, homogeneous_xyz.transpose(N, M)
    )
    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
        return pc
    return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)

class SO3:
    """
    A generic class defining a 3D orientation. Mostly a wrapper around quaternions
    """

    def __init__(self, quaternion):
        """
        :param quaternion: Quaternion
        """
        if isinstance(quaternion, Quaternion):
            self._quat = quaternion
        elif isinstance(quaternion, (np.ndarray, list)):
            self._quat = Quaternion(np.asarray(quaternion))
        else:
            raise Exception("Input to SO3 must be Quaternion, np.ndarray, or list")

    def __repr__(self):
        return f"SO3(quaternion={self.wxyz})"

    @classmethod
    def from_rpy(cls, r, p, y):
        """
        Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

        The roll-pitch-yaw axes in a typical URDF are defined as a
        rotation of ``r`` radians around the x-axis followed by a rotation of
        ``p`` radians around the y-axis followed by a rotation of ``y`` radians
        around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
        Wikipedia_ for more information.

        .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

        :param rpy: The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
        :return: An SO3 object
        """
        c3, c2, c1 = np.cos([r, p, y])
        s3, s2, s1 = np.sin([r, p, y])

        matrix = np.array(
            [
                [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
                [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
                [-s2, c2 * s3, c2 * c3],
            ],
            dtype=np.float64,
        )
        return cls(Quaternion(matrix=matrix))

    @classmethod
    def from_unit_axes(cls, x, y, z):
        assert np.isclose(np.dot(x, y), 0)
        assert np.isclose(np.dot(x, z), 0)
        assert np.isclose(np.dot(y, z), 0)
        assert np.isclose(np.linalg.norm(x), 1)
        assert np.isclose(np.linalg.norm(y), 1)
        assert np.isclose(np.linalg.norm(z), 1)
        m = np.eye(4)
        m[:3, 0] = x
        m[:3, 1] = y
        m[:3, 2] = z
        return cls(Quaternion(matrix=m))

    @property
    def inverse(self):
        """
        :return: The inverse of the orientation
        """
        return SO3(self._quat.inverse)

    @property
    def rpy(self):
        """
        This might not be the most numerically stable and should probably be replaced
        by whatever Eigen has
        """
        matrix = self.matrix
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
        pitch = np.arctan2(
            -matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2)
        )
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        return roll, pitch, yaw

    @property
    def transformation_matrix(self):
        return self._quat.transformation_matrix

    @property
    def xyzw(self):
        """
        :return: A list representation of the quaternion as xyzw
        """
        return self._quat.vector.tolist() + [self._quat.scalar]

    @property
    def wxyz(self):
        """
        :return: A list representation of the quaternion as wxyz
        """
        return [self._quat.scalar] + self._quat.vector.tolist()

    @property
    def matrix(self):
        """
        :return: The matrix representation of the orientation
        """
        return self._quat.rotation_matrix

class SE3:
    """
    A generic class defining a 3D pose with some helper functions for easy conversions
    """

    def __init__(self, matrix=None, xyz=None, quaternion=None, so3=None, rpy=None):
        assert bool(matrix is None) != bool(
            xyz is None
            and (bool(quaternion is None) ^ bool(so3 is None) ^ bool(rpy is None))
        )
        if matrix is not None:
            self._xyz = matrix[:3, 3]
            self._so3 = SO3(Quaternion(matrix=matrix, rtol=1e-03, atol=1e-03))
        else:
            self._xyz = np.asarray(xyz)
            if quaternion is not None:
                self._so3 = SO3(quaternion)
            elif rpy is not None:
                self._so3 = SO3.from_rpy(*rpy)
            else:
                self._so3 = so3

    def __repr__(self):
        return f"SE3(xyz={self.xyz}, quaternion={self.so3.wxyz})"

    def __matmul__(self, other):
        """
        Allows for numpy-style matrix multiplication using `@`
        """
        return SE3(matrix=self.matrix @ other.matrix)

    @property
    def inverse(self):
        """
        :return: The inverse transformation
        """
        so3 = self._so3.inverse
        xyz = -so3.matrix @ self._xyz
        return SE3(xyz=xyz, so3=so3)

    @property
    def matrix(self):
        """
        :return: The internal matrix representation
        """
        m = self._so3.transformation_matrix
        m[:3, 3] = self.xyz
        return m

    @property
    def so3(self):
        """
        :return: The representation of orientation
        """
        return self._so3

    @so3.setter
    def so3(self, val):
        """
        :param val: A pose object
        """
        assert isinstance(val, SO3)
        self._so3 = val

    @property
    def xyz(self):
        """
        :return: The translation vector
        """
        return self._xyz.tolist()

    @xyz.setter
    def xyz(self, val):
        """
        :return: The translation vector
        """
        self._xyz = np.asarray(val)

    @classmethod
    def from_unit_axes(cls, origin, x, y, z):
        """
        Constructs SE3 object from unit axes indicating direction and an origin

        :param origin: np.array indicating the placement of the origin
        :param x: A unit axis indicating the direction of the x axis
        :param y: A unit axis indicating the direction of the y axis
        :param z: A unit axis indicating the direction of the z axis
        :return: SE3 object
        """
        so3 = SO3.from_unit_axes(x, y, z)
        return cls(xyz=origin, so3=so3)