from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Set, Union

# pylint: disable=cyclic-import
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import (
    get_prim_at_path,
    get_prim_children,
    get_prim_path,
    is_prim_path_valid,
)
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# pylint: disable=cyclic-import
from tongverse.object.base_object import BaseObject
from tongverse.object.rigid_object import RigidObject
from tongverse.utils.constant import TmpPath


class ObjectStates(ABC):
    """Abstract base class for defining object states."""

    def __init__(self, obj: RigidObject) -> None:
        """
        Initialize ObjectStates.

        Args:
            obj (RigidObject): The object to associate with the state.
        """
        if not isinstance(obj, RigidObject):
            raise TypeError(f"{obj} is not a Rigid Object")

        self.obj = obj

    @abstractmethod
    def get_value(self) -> Union[bool, Set[str]]:
        """Abstract method to retrieve the value of the state."""


class AbsoluteObjectStates(ObjectStates):
    """Abstract class for absolute object states."""

    @abstractmethod
    def get_value(self) -> Union[bool, Set[str]]:
        """Abstract method to retrieve the value of the absolute state."""


class RelativeObjectStates(ObjectStates):
    """Abstract class for relative object states."""

    @abstractmethod
    # pylint: disable=arguments-differ
    def get_value(
        self, other_obj: Optional[BaseObject] = None
    ) -> Union[bool, Set[str]]:
        """
        Abstract method to retrieve the value of the relative state.

        Args:
            other_obj (Optional[BaseObject]): The other object to compare with.

        Returns:
            Union[bool, Set[str]]: The value of the relative state.
        """


class Touching(RelativeObjectStates):
    """State representing whether an object is touching another object."""

    def get_value(self, other_obj: Optional[BaseObject] = None) -> bool:
        """
        Get the value of the touching state.

        Args:
            other_obj (Optional[BaseObject]): The other object to check for touching.

        Returns:
            bool: True if touching, False otherwise.
        """
        if other_obj is not None and not isinstance(other_obj, BaseObject):
            raise TypeError(f"args {other_obj} should either None or Base Object type")
        current_contacts = self.obj.get_current_contacts()
        if len(current_contacts) > 0:
            if other_obj is None or other_obj.name in current_contacts:
                return True
        return False


class SupportedBy(RelativeObjectStates):
    """State representing whether an object is supported by another object."""

    def get_value(self, other_obj: Optional[BaseObject] = None) -> bool:
        """
        Get the value of the supported by state.

        Args:
            other_obj (Optional[BaseObject]): The supporting object.

        Returns:
            bool: True if supported, False otherwise.
        """
        if other_obj is None or not isinstance(other_obj, BaseObject):
            raise TypeError(f"args {other_obj} should be Base Object type")
        current_contacts = self.obj.get_current_contacts()
        current_z_pos = self.obj.get_world_pose()[0][-1]

        if len(current_contacts) == 0:
            return False

        if other_obj is not None and other_obj.name in current_contacts:
            return other_obj.get_world_pose()[0][-1].item() <= current_z_pos.item()

        return False


class Inside(RelativeObjectStates):
    """State representing whether an object is inside another object."""

    def get_value(self, other_obj: Optional[BaseObject] = None) -> bool:
        """
        Get the value of the inside state.

        Args:
            other_obj (Optional[BaseObject]): The other object to check for containment.

        Returns:
            bool: True if inside, False otherwise.
        """
        if other_obj is not None and not isinstance(other_obj, BaseObject):
            raise TypeError(f"args {other_obj} should either None or Base Object type")
        horizontal_nexts = self.obj.horizontal_next_to()
        vertical_nexts = self.obj.vertical_next_to()
        intersets = horizontal_nexts & vertical_nexts
        if len(intersets) == 1 and None in intersets:
            return False

        if other_obj is None and len(intersets) > 0:
            return True

        if len(intersets) > 0 and other_obj.name in intersets:
            return True

        return False


class AttachedTo(RelativeObjectStates):
    """This class manages the attachment between two rigid bodies by establishing a
    fixed joint between the child(self.obj) and parent bodies(other obj).

    NOTE:
    1. The fixed joint remains inactive when the timeline is stopped, resulting in
        the bodies it connects only snapping together when the simulation is running.
    2. While a parent body can have multiple child body, child body can only have
        one parent body.
    """

    def get_value(self, other_obj: Optional[Union[RigidObject, RigidPrim]] = None):
        """
        Check if self.obj is attached to other object.

        Args:
            other_obj (Optional[BaseObject]): The other object to check for attachment.
            If other_obj is None, check all possible tmp fixed joint

        Returns:
            bool: True if being attached, False otherwise.
        """
        tmp_fixed_joint_prefix_path = f"{TmpPath.FIXED_JOINT_PREFIX}"
        if not is_prim_path_valid(tmp_fixed_joint_prefix_path):
            return False

        if other_obj is None:
            fixed_joint_prefix_prim = get_prim_at_path(tmp_fixed_joint_prefix_path)
            for child_prim in get_prim_children(fixed_joint_prefix_prim):
                if child_prim.GetName().find(self.obj.name) == 0:
                    return True
            return False

        fixed_joint_prim = (
            f"{TmpPath.FIXED_JOINT_PREFIX}/{self.obj.name}_{other_obj.name}"
        )
        if not is_prim_path_valid(fixed_joint_prim):
            return False
        return True

    def set_value(
        self, other_obj: Optional[Union[RigidObject, RigidPrim]], keep_pose: bool = True
    ) -> bool:
        # Check if self.obj is already being attached
        if self.get_value():
            return False
        child_body = self.obj.baselink.prim
        # Check if other_obj is rigid.
        if isinstance(other_obj, RigidObject):
            parent_body = other_obj.baselink.prim
            self._create_fix_joint(parent_body, child_body, keep_pose)
            return True
        if isinstance(other_obj, RigidPrim):
            parent_body = other_obj.prim
            self._create_fix_joint(parent_body, child_body, keep_pose)
            return True

        # Return False if other_ob is not rigid
        return False

    def _create_fix_joint(
        self, parent_body: Usd.Prim, child_body: Usd.Prim, keep_pose: bool = True
    ):
        """
        Args:
            parent_body (Usd.Prim): it can be robot's body parent or object's root_link.
            child_body (Usd.Prim): self.obj's baselink prim
            keep_pose: keep the origin distance and position between body1 and body0,
            in this case, no need to specify relative position and quaternion.


        """
        # Create the joint
        joint_path = (
            f"{TmpPath.FIXED_JOINT_PREFIX}/{self.obj.name}_{parent_body.GetName()}"
        )

        path0 = get_prim_path(parent_body)
        path1 = get_prim_path(child_body)

        fixed_joint = UsdPhysics.FixedJoint.Define(get_current_stage(), joint_path)
        fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path(path0)])
        fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(path1)])

        if keep_pose:
            xf_cache = UsdGeom.XformCache()
            child_pose = xf_cache.GetLocalToWorldTransform(child_body)
            parent_pose = xf_cache.GetLocalToWorldTransform(parent_body)

            rel_pose = child_pose * parent_pose.GetInverse()
            rel_pose = rel_pose.RemoveScaleShear()
            pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
            rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())

            fixed_joint.CreateLocalPos0Attr().Set(pos1)
            fixed_joint.CreateLocalRot0Attr().Set(rot1)
            fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        return fixed_joint


_DEFAULT_OBJECT_STATES = set([Touching, SupportedBy, Inside, AttachedTo])
