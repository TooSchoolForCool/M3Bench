from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import RigidPrim
from omni.isaac.dynamic_control import _dynamic_control

# pylint: enable=wrong-import-position
from tongverse.object.rigid_object import RigidObject


class ArticulatedObject(RigidObject):
    """A class representing an articulated object, applying Articulation API to
    RigidObject.

    NOTE:
        1. The ArticulatedObject instance is initialized during scene loading.
        2. In dynamic control, the orientation format is (QX, QY, QZ, QW),
            which is different from the orientation format obtained from Core (
                QW, QX, QY, QZ).
        3. Some articulated objects have fixed joints linked to the world.
            Therefore, avoid using the omni.core.articulation API to traverse joint,
            as it may return the same articulation handle (world as root).
            Instead, use dynamic control API to traverse parent/child joints.
        4. The articulation must be physics initialized, as well as the prim physics
           initialized, in order to operate on it.
        See the `initialize` method for more details.
        5. self._art.articulation_handle is a unique identifier used by the Dynamic
           Control extension to manage the articulation
        6.  Uses a dictionary called 'links_dict' to store part link prim.
            Key represents the link name, and the value corresponds to the link prim.
        7.  Uses a dictionary called 'parts_info_dict' to store part link information.
            Key represents the link name, and the value corresponds to its symbolic
            info.
    """

    def __init__(self, obj_cfg: dict, path_prefix="/World/Scene") -> None:
        """Initialize an ArticulatedObject instance.

        Parameters:
            path_prefix (str, optional): The prefix for the path to the object
                   in the scene hierarchy. Defaults to "/World/Scene".
            obj_cfg (dict): The configuration for the object.
                   This information is obtained after processing the metadata file.

        Raises:
            TypeError: If obj_cfg is None or not a dictionary.
        """
        if obj_cfg is None or not isinstance(obj_cfg, dict):
            raise TypeError("obj_cfg must be a dictionary")
        # Articulation hanlde
        self._art = 0
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._joint_names = set()
        # baselink prim body handle(from dynamic control)
        self._baselink_body_handle = 0
        # Indicates whether there is a fixed joint linked to the world
        self._fixed = False
        self._links_dict = {}
        self._parts_info_dict = {}
        super().__init__(obj_cfg, path_prefix)

    def _process_cfgs(self) -> None:
        """Process the configurations.
        It creates rigid primfor each link specified in the
        configuration and adds them to the `links_dict`.."""
        super()._process_cfgs()
        for link_name in self._obj_cfg.get("link"):
            self._links_dict[link_name] = RigidPrim(
                f"{self._path_prefix}/{link_name}", self._obj_name
            )

        self._art = Articulation(self.baselink_path)
        self._fixed = self._obj_cfg.get("fixed_to_world", False)
        self._parts_info_dict = self._obj_cfg.get("part_info", None)

    def _post_process(self) -> None:
        """This method should be called after the initialize_physics() method
        has been executed.

        This method stores joint information by using the dynamic control API
        to iterate through the child joints of the object and checks if it
        has a fixed joint linked to the world.

        NOTE: Setting velocity or enabling/disabling gravity doesn't work on
               objects with fixed joints linked to the world.
        """
        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        cnt = self._dc.get_rigid_body_child_joint_count(self._baselink_body_handle)
        self._joint_names = set()
        for i in range(cnt):
            joint_handle = self._dc.get_rigid_body_child_joint(
                self._baselink_body_handle, i
            )
            # Currently, only joints with one degree of freedom are supported
            if self._dc.get_joint_dof_count(joint_handle) == 1:
                name = self._dc.get_joint_name(joint_handle)
                self._joint_names.add(name)

    def initialize_physics(self):
        """
        Initializes physics for the articulation and set to the default states

        This method needs to be called after each hard reset
        (e.g., Stop + Play on the timeline)
        before interacting with any other class method.
        """
        if self._physics_disabled is True:
            self.disable_physics(False)
        if self._disable_gravity is True:
            self.disable_gravity(False)

        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._joint_names = set()
        for link in self._links_dict.values():
            link.initialize()
            link.post_reset()

        self._art.initialize()
        self._dc.wake_up_articulation(self._art.articulation_handle)
        self._baselink_body_handle = self._dc.get_rigid_body(self.baselink_path)
        self._post_process()

    def set_world_pose(
        self,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Sets the world pose of the object.

        Parameters:
            position (Sequence[float], optional): The position coordinates.
            Defaults to None.
            orientation (Sequence[float], optional): The orientation coordinates.
            Defaults to None.

        Raises:
            Exception: If the ArticulatedObject is not initialized with physics.
            Call env.reset() first.
        """
        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics."
                "Call env.reset() first."
            )
        return self._art.set_world_pose(position, orientation)

    def set_local_pose(
        self,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Sets the local pose of the object.

        Parameters:
            translation (Sequence[float], optional): The translation coordinates.
            Defaults to None.
            orientation (Sequence[float], optional): The orientation coordinates.
            Defaults to None.

        Raises:
            RuntimeError: If the ArticulatedObject is not initialized with physics.
            Call env.reset() first.
        """
        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        return self._art.set_local_pose(translation, orientation)

    def get_world_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the world pose of the object root.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the position tensor and
            the orientation tensor.

        Raises:
            RuntimeError: If the ArticulatedObject is not initialized with physics.
            Call env.reset() first.
        """
        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        return self._art.get_world_pose()

    def get_local_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the local pose of the object root.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the position tensor
            and the orientation tensor.

        Raises:
            RuntimeError: If the ArticulatedObject is not initialized with physics.
            Call env.reset() first.
        """
        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        return self._art.get_local_pose()

    def get_joint_positions(self, joint_name: Optional[str] = None) -> dict:
        """
        Gets the positions of the joints.

        Args:
            joint_name (str, optional): The name of the joint.
            If None, returns positions of all joints of this object.

        Returns:
            Dict[str, float]: A dictionary containing joint names as keys and
            their corresponding positions as values.

        Raises:
            RuntimeError: If the ArticulatedObject is not initialized with physics.
            Call env.reset() first.
            ValueError: If the specified joint does not exist in this object.
        """
        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics."
                "Call env.reset() first."
            )

        res = {}
        if joint_name is not None:
            if joint_name not in self._joint_names:
                raise ValueError(
                    f"The joint '{joint_name}' "
                    f"does not exist in ArticulatedObject {self.name}."
                )
            dof_ptr = self._dc.find_articulation_dof(
                self._art.articulation_handle, joint_name
            )
            dof_state = self._dc.get_dof_state(dof_ptr, _dynamic_control.STATE_ALL)
            return {joint_name: dof_state.pos}

        for joint_name in self._joint_names:  # pylint: disable=R1704
            dof_ptr = self._dc.find_articulation_dof(
                self._art.articulation_handle, joint_name
            )
            dof_state = self._dc.get_dof_state(dof_ptr, _dynamic_control.STATE_ALL)
            res[joint_name] = dof_state.pos

        return res

    def set_joint_positions(self, joint_name: str, pos_val: float) -> None:
        """Set the position of a joint in the articulated object.

        Args:
            joint_name (str): The name of the joint.
            pos_val (float): The position value to set for the joint.

        Raises:
            RuntimeError: If the articulated object is not initialized with physics.
            ValueError: If the specified joint does not exist in the articulated object.
        """
        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        if joint_name is not None:
            if joint_name not in self._joint_names:
                raise ValueError(
                    f"The joint '{joint_name}' does not exist "
                    f"in ArticulatedObject {self.name}."
                )

            dof_ptr = self._dc.find_articulation_dof(
                self._art.articulation_handle, joint_name
            )
            self._dc.set_dof_position(dof_ptr, pos_val)

    def disable_physics(self, set_value: bool):
        """
        NOTE: @QI
        1. can't call disable_rigid_body_physics() with object
        who has joints, since eDISABLE_SIMULATION is only supported by
        PxRigidStatic and PxRigidDynamic actors
        As a work around, we put the articulation to sleep
        2. can't set kinematic with object who has joints,
        Warning "CreateJoint - cannot create a joint between static bodies"
        """
        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )

        if set_value:
            self._dc.sleep_articulation(self._art.articulation_handle)
            self._dc.sleep_rigid_body(self._baselink_body_handle)
            self._physics_disabled = True
        else:
            # Enable physics for a articulation
            self._dc.wake_up_articulation(self._art.articulation_handle)
            # Enable physics for a rigid body
            self._dc.wake_up_rigid_body(self._baselink_body_handle)
            self._physics_disabled = False

    def keep_still(self) -> None:
        """keep the articulated object still by setting its angular and linear
        velocities and it's joints velocities to zero.
        """
        self.set_angular_velocity(torch.zeros(3))
        self.set_linear_velocity(torch.zeros(3))
        # set all joints velocity to zero
        for joint_name in self._joint_names:  # pylint: disable=R1704
            dof_ptr = self._dc.find_articulation_dof(
                self._art.articulation_handle, joint_name
            )
            self._dc.set_dof_velocity(dof_ptr, 0)

    def disable_gravity(self, set_value: bool) -> None:
        """Disable gravity for the articulated object.
        enable or disable gravity doesn't work on those objects who has fixed
        joint linked to the world
        """
        if self._fixed:
            return

        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        if set_value:
            self._art.disable_gravity()
            self._disable_gravity = True
        else:
            self._art.enable_gravity()
            self._disable_gravity = False

    def set_angular_velocity(self, velocity: torch.Tensor) -> None:
        """
        Set the angular velocity of the root articulation link
        Args:
            velocity (torch.Tensor)): 3D angular velocity vector. Shape (3,)
        Raises:
            RuntimeError: If the articulated object is not initialized with physics.
        """
        if self._fixed:
            return

        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        if self._fixed:
            return
        self._art.set_angular_velocity(velocity)

    def get_angular_velocity(self) -> torch.Tensor:
        """
        Get the angular velocity of the root articulation link.

        Returns:
            torch.Tensor: Current angular velocity of the rigid link. Shape (3,).

        Raises:
            RuntimeError: If the articulated object is not initialized with physics.
        """
        if self._fixed:
            return torch.zeros(3)

        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        return self._art.get_angular_velocity()

    def set_linear_velocity(self, velocity: torch.Tensor) -> None:
        """
        Set the linear velocity of the root articulation link.

        Args:
            velocity (torch.Tensor): 3D linear velocity vector. Shape (3,).

        Raises:
            RuntimeError: If the articulated object is not initialized with physics.

        NOTE:
            Objects with fixed joints linked to the world, velocities will remain zero
            even after calling the set_linear_velocity method
        """
        if self._fixed:
            return

        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        self._art.set_linear_velocity(velocity)

    def get_linear_velocity(self) -> torch.Tensor:
        """
        Get the linear velocity of the root articulation link.

        Returns:
            torch.Tensor: Current linear velocity of the rigid link. Shape (3,).

        Raises:
            RuntimeError: If the articulated object is not initialized with physics.

        NOTE:
            Objects with fixed joints linked to the world, velocities will remain zero
            even after calling the set_linear_velocity method
        """
        if self._fixed:
            return torch.zeros(3)

        if _dynamic_control.INVALID_HANDLE in [
            self._baselink_body_handle or self._art.articulation_handle
        ]:
            raise RuntimeError(
                f"ArticulatedObject {self.name} must be initialized with physics. "
                "Call env.reset() first."
            )
        return self._art.get_linear_velocity()

    @property
    def joint_names(self) -> set:
        return self._joint_names

    @property
    def parts_info(self) -> dict:
        return self._parts_info_dict

    @property
    def links_dict(self) -> dict:
        """
        Retrieves the dictionary containing the links of the object.

        Returns:
            dict: A dictionary where the keys are the link names and the values
            are the link prims.
        """
        return self._links_dict

    def set_gain(self):
        pass

    def get_gain(self):
        pass

    def get_joint_limit(self):
        pass

    def set_joint_limit(self):
        pass
