from __future__ import annotations

from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdSkel


class AnimationInterface:
    r"""Interface for Skeleton Animation that controls the skeleton in USD."""

    def __init__(self, prim_path):
        """Initialize the AnimationPrim.

        Args:
            prim_path (str): Path to the prim in stage.
        """
        stage = get_current_stage()
        self._animation: UsdSkel.Animation = UsdSkel.Animation.Define(stage, prim_path)

        # super().__init__(prim_path)

        self._animation_query: UsdSkel.AnimQuery = UsdSkel.Cache().GetAnimQuery(
            self._animation
        )
        self._joint_names = []

    def get_joints(self):
        """Get the joints to be animated.

        Returns:
            list: List of joint paths to be animated.
        """
        return self._animation.GetJointsAttr().Get()

    def set_joints(self, joints):
        """Set the joints to be animated.

        Args:
            joints (list): List of joint paths to be animated.
        """
        self._animation.GetJointsAttr().Set(joints)
        self._joint_names = [joint.split("/")[-1] for joint in joints]

    def set_local_orientations(self, orientations):
        """Set the local orientations of the joints.

        Args:
            orientations (list): List of orientations in the format (w, x, y, z).
                The length of the list should be the same as the number of joints.
                The order of the orientations should match the order of the joints.
        """
        transforms_array = self.get_local_transforms()
        # Directly iterate over the transforms and orientations to avoid the need to
        # find the index of the joint, reducing the complexity
        for i, mat_orient in enumerate(
            zip(transforms_array, orientations, strict=False)
        ):
            mat, orient = mat_orient
            w, x, y, z = orient
            transform = Gf.Transform(mat)
            transform.SetRotation(Gf.Rotation(Gf.Quaternion(w, Gf.Vec3d(x, y, z))))
            transforms_array[i] = transform.GetMatrix()
        self._animation.SetTransforms(xforms=transforms_array)

    def set_local_orientations_by_names(self, orientations, names):
        """Set the local orientations of the joints by names.

        Args:
            orientations (list): List of orientations in the format (w, x, y, z).
                The length of the list should be the same as the number of names.
                The order of the orientations should match the order of the names.
            names (list): List of joint names to set the orientations.
        """
        transforms_array = self.get_local_transforms()
        for orient, name in zip(orientations, names, strict=False):
            index = self._joint_names.index(name)
            w, x, y, z = orient
            transform = Gf.Transform(transforms_array[index])
            transform.SetRotation(Gf.Rotation(Gf.Quaternion(w, Gf.Vec3d(x, y, z))))
            transforms_array[index] = transform.GetMatrix()
        self._animation.SetTransforms(xforms=transforms_array)

    def set_local_transforms(self, xforms):
        """Convenience method for setting an array of transforms. The given transforms
        must be *orthogonal*.

        Args:
            xforms (list): List of transforms to set.
        """
        self._animation.SetTransforms(xforms)

    def set_local_transforms_by_names(self, xforms, names):
        """Set the local transforms of the joints by names.

        Args:
            xforms (list): List of transforms to set.
            names (list): List of joint names to set the transforms.
        """
        indexes = [self._joint_names.index(name) for name in names]
        current_transforms = self._animation.GetTransforms()
        for index, xform in zip(indexes, xforms, strict=False):
            current_transforms[index] = xform
        self._animation.SetTransforms(current_transforms)

    def get_local_transforms(self):
        """Convenience method for getting an array of transforms. The returned
        transforms are *orthogonal*.

        Returns:
            Matrix4dArray: List of transforms.
        """
        return self._animation.GetTransforms()

    def get_local_transforms_by_names(self, names):
        """Get the local transforms of the joints by names.

        Args:
            names (list): List of joint names to get the transforms.

        Returns:
            Matrix4dArray: List of transforms.
        """
        indexes = [self._joint_names.index(name) for name in names]
        current_transforms = self._animation.GetTransforms()
        transforms = [current_transforms[index] for index in indexes]
        return transforms
