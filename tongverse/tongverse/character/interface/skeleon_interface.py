from __future__ import annotations

from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdSkel


class SkeletonInterface:
    r"""Interface for Skeletons in USD."""

    def __init__(self, prim_path):
        """Initialize the SkeletonInterface.

        Args:
            prim_path (str): Path to the prim in stage.
        """
        stage = get_current_stage()
        self._skeleton: UsdSkel.Skeleton = UsdSkel.Skeleton.Get(stage, prim_path)
        self._skeleton_query: UsdSkel.SkeletonQuery = UsdSkel.Cache().GetSkelQuery(
            self._skeleton
        )
        self._binding_api: UsdSkel.BindingAPI = UsdSkel.BindingAPI.Apply(
            self._skeleton.GetPrim()
        )

    def get_joint_order(self):
        """Get the order of the joints in the skeleton.

        Returns:
            list: List of joint paths in the skeleton. The path of each joint is the
                topological path from the root of the skeleton to the joint.
        """
        return self._skeleton_query.GetJointOrder()

    def get_joint_names(self):
        """Get the names of the joints in the skeleton.

        Returns:
            list: List of joint names in the skeleton. The name of each joint does not
                include the topological path from the root of the skeleton to the joint.
        """
        joints = self._skeleton_query.GetJointOrder()
        names = [joint.split("/")[-1] for joint in joints]
        # names = [Sdf.Path(joint).name for joint in joints]
        return names

    def get_rest_joint_local_transforms(self):
        """Get the rest local transforms of the joints in the skeleton.

        Returns:
            list: List of rest local transforms of the joints in the skeleton.
        """
        return self._skeleton.GetRestTransformsAttr().Get()

    def set_rest_joint_local_transforms(self, rest_transforms):
        """Set the rest local transforms of the joints in the skeleton.

        Args:
            rest_transforms (list): List of rest local transforms of the joints in the
                skeleton. The length of the list should be the same as the number of
                joints in the skeleton. Each element in the list should be a 4x4 matrix
                representing the rest local transform of the corresponding joint.
        """
        # rest_transforms = Vt.Matrix4dArray().FromNumpy(rest_transforms)
        self._skeleton.GetRestTransformsAttr().Set(rest_transforms)

    def set_animation_target(self, target_animation_path):
        """Set the target animation for the skeleton.

        Args:
            target_animation_path (str): Path to the target animation in stage.
        """
        self._binding_api.CreateAnimationSourceRel().SetTargets([target_animation_path])
