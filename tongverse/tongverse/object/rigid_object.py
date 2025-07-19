from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple

import torch
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.sensor import _sensor
from omni.physx import get_physx_scene_query_interface
from pxr import PhysxSchema, UsdGeom

# pylint: enable=wrong-import-position
from tongverse.object.base_object import BaseObject


class RigidObject(BaseObject):
    """Applies rigidbody API API to BaseObject."""

    def __init__(self, obj_cfg: dict, path_prefix="/World/Scene") -> None:
        """
        Initializes a RigidObject instance during scene loading.

        Parameters:
            path_prefix (str, optional): The prefix for the path to the object in
            the scene hierarchy. Defaults to "/World/Scene".
            obj_cfg (dict): The configuration for the object. This information is
            obtained after processing the metadata file.

        Raises:
            TypeError: If obj_cfg is None or not a dictionary.

        NOTE:
            When a continuous motion is not desired,self. kinematic flag
            should be set to false.

        """

        if obj_cfg is None or not isinstance(obj_cfg, dict):
            raise TypeError("obj_cfg must be a dictionary")

        self._cs = _sensor.acquire_contact_sensor_interface()
        self._states = {}
        super().__init__(obj_cfg, path_prefix)
        PhysxSchema.PhysxContactReportAPI.Apply(self.baselink.prim)
        # NOTE:  somehow contact sensor have to be called first aftrer created
        self._cs.get_rigid_body_raw_data(self.baselink_path)
        self._physics_disabled = False
        self._disable_gravity = False

    def initialize_physics(self):
        """
        Initializes physics for the object.
        """
        if self._physics_disabled is True:
            self.disable_physics(False)
        if self._disable_gravity is True:
            self.disable_gravity(False)

        super().initialize_physics()

    def disable_gravity(self, set_value: bool) -> None:
        """Enable/Disable gravity on rigid bodies (enabled by default)."""
        if set_value:
            self._disable_gravity = True
            self.baselink._rigid_prim_view.disable_gravities()  # pylint: disable=W0212
        else:
            self._disable_gravity = False
            self.baselink._rigid_prim_view.enable_gravities()  # pylint: disable=W0212

    def disable_physics(self, set_value: bool) -> None:
        """A kinematic body is a body that moved through animated poses or through
        user defined poses.
        NOTE:
            When rigid body physics are disabled:
            - Velocities and wake counter are set to 0.
            - All forces are cleared, and kinematic targets are reset.
            - The actor is put to sleep, and touching actors from the
            previous frame are awakened.

            Additionally, the following calls are forbidden:
            - PxRigidBody: setLinearVelocity(), setAngularVelocity(), addForce(),
            addTorque(), clearForce(), clearTorque(), setForceAndTorque()
            - PxRigidDynamic: setKinematicTarget(), setWakeCounter(), wakeUp(),
            putToSleep()
        """
        if set_value:
            self.baselink.prim.GetAttribute("physics:kinematicEnabled").Set(True)
            self._physics_disabled = True
        else:
            self.baselink.prim.GetAttribute("physics:kinematicEnabled").Set(False)
            self._physics_disabled = False

    def get_mass(self) -> float:
        """Get the mass of the object

        Returns:
            float: mass of the object in kg.
        """

        return self.baselink.get_mass()

    def set_mass(self, mass: float) -> None:
        """Set the mass of the object

        Args:
            mass (float): mass of the obj in kg.
        """
        self.baselink.set_mass(mass)

    def get_density(self) -> float:
        """
        Retrieves the density of the object.
        """
        return self.baselink.get_density()

    def set_density(self, density: float) -> None:
        """Set the density of the obj

        Args:
            density (float): density of obj
        """
        self.baselink.set_density(density)

    def set_angular_velocity(self, velocity: torch.Tensor) -> None:
        """
        Sets the angular velocity of the object.

        Args:
            velocity (torch.Tensor)): The 3D angular velocity vector.

        Raises:
            RuntimeError: If the object's rigid body physics is disabled.
        """
        if self._physics_disabled is True:
            raise RuntimeError(
                f"Setting the angular velocity is not possible, "
                f"{self.name} must be non-kinematic."
            )
        self.baselink.set_angular_velocity(velocity)

    def get_angular_velocity(self) -> torch.Tensor:
        """
        Retrieves the angular velocity of the object.

        Returns:
            torch.Tensor: The current angular velocity of the object.
        """
        return self.baselink.get_angular_velocity()

    def set_linear_velocity(self, velocity: torch.Tensor) -> None:
        """
        Sets the linear velocity of the object.

        Args:
            velocity (torch.Tensor): The 3D linear velocity vector.

        Raises:
            RuntimeError: If the object's rigid body physics is disabled.
        """
        if self._physics_disabled is True:
            raise RuntimeError(
                f"Setting the linear velocity is not possible, "
                f"{self.name} must be non-kinematic."
            )
        self.baselink.set_linear_velocity(velocity)

    def get_linear_velocity(self) -> torch.Tensor:
        """
        Retrieves the linear velocity of the object.

        Returns:
            torch.Tensor: The current linear velocity of the object.
        """
        return self.baselink.get_linear_velocity()

    def set_world_pose(
        self,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ):
        """
        Sets the world pose of the object.

        Args:
            position (Optional[Sequence[float]]): The position to set.
            orientation (Optional[Sequence[float]]): The orientation to set.

        NOTE: Cannot assign transform to non-root articulation link
        """
        if position is not None:
            self.baselink.set_world_pose(position=position)
        if orientation is not None:
            self.baselink.set_world_pose(orientation=orientation)

    def set_local_pose(
        self,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Sets the local pose of the object.

        Args:
            translation (Optional[Sequence[float]]): The translation to set.
            orientation (Optional[Sequence[float]]): The orientation to set.
        """
        if translation is not None:
            self.baselink.set_local_pose(translation=translation)

        if orientation is not None:
            self.baselink.set_local_pose(orientation=orientation)

    def get_current_contacts(self) -> set[str]:
        """
        NOTE: only work when turn on simulation
        Returns:
            set[str]: list of all current contacts objects names
        """
        contacts = set()
        raw_data = self._cs.get_rigid_body_raw_data(self.baselink_path)

        for c in raw_data:
            obj1 = self._cs.decode_body_name(c["body0"]).split("/")[-1]
            obj2 = self._cs.decode_body_name(c["body1"]).split("/")[-1]
            contacts.add(obj1)
            contacts.add(obj2)

        # remove itself
        if self.name in contacts:
            contacts.remove(self.name)
        return contacts

    def get_aabb(self) -> Tuple[float, float, float, float, float, float]:
        """
        Returns:
            Tuple[float, float, float, float, float, float]: Bounding box for this prim,
            [min x, min y, min z],[max x, max y, max z]
        """
        cache = create_bbox_cache()
        min_x, min_y, min_z, max_x, max_y, max_z = compute_aabb(
            cache, self.baselink_path
        )
        return min_x, min_y, min_z, max_x, max_y, max_z

    def check_raycast(
        self, origin: Sequence[float], ray_dir: Sequence[float], max_dist: float = 5
    ) -> Optional[str]:
        """
        Detects the closest object that intersects with a specified ray

        Parameters:
            origin (Sequence[float]): The origin point of the ray in 3D space.
            ray_dir (Sequence[float]): The direction vector of the ray.
            max_dist (float, optional): The maximum distance to check along the ray.
            Defaults to 5.

        Returns:
            Optional[str]: The name of the closest hit object in the specified
            direction, or None if no object is hit.

        NOTE:
            The following is assumed: the stage contains a physics scene,
            all objects have collision meshes enabled, and simulation is
            being turned on.
        """
        # closest touched object
        stage = get_current_stage()
        # physX query to detect closest hit
        hit = get_physx_scene_query_interface().raycast_closest(
            origin, ray_dir, max_dist
        )
        if hit["hit"]:
            # Change object color to yellow and record distance from origin
            usd_geom = UsdGeom.Mesh.Get(stage, hit["rigidBody"])
            return usd_geom.GetPath().pathString.split("/")[-1]
        return None

    def horizontal_next_to(self, offset: float = 0.01) -> Set[Optional[str]]:
        """
        Computes objects that are horizontally adjacent to the object.

        Args:
            offset (float, optional): The offset used in raycasting to detect adjacency.
            Defaults to 0.01.

        Returns:
            Set[Optional[str]]: A set containing the names of objects that are
            horizontally adjacent to the current object. If no object is adjacent
            in a particular direction, the corresponding entry in the set will be None.
        """
        min_x, min_y, min_z, max_x, max_y, max_z = self.get_aabb()
        directions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        horizontal_nexts = set()

        for d in directions:
            for corner in [
                (max_x, max_y, max_z - offset),
                (min_x, max_y, max_z - offset),
                (max_x, max_y, min_z + offset),
                (max_x, min_y, max_z - offset),
                (min_x, min_y, min_z + offset),
                (min_x, min_y, max_z - offset),
                (max_x, min_y, min_z + offset),
                (min_x, max_y, min_z + offset),
            ]:
                obj = self.check_raycast(corner, d)
                horizontal_nexts.add(obj)

        return horizontal_nexts

    def vertical_next_to(self) -> Set[Optional[str]]:
        """
        Computes objects that are vertically adjacent to the object.

        Returns:
            Set[Optional[str]]: A set containing the names of objects that are
            vertically adjacent to the current object. If no object is adjacent
            in a particular direction, the corresponding entry in the set will be None.
        """

        min_x, min_y, min_z, max_x, max_y, max_z = self.get_aabb()
        directions = [[0, 0, 1], [0, 0, -1]]
        vertical_nexts = set()
        for d in directions:
            for corner in [
                (max_x, max_y, max_z),
                (min_x, max_y, max_z),
                (max_x, max_y, min_z),
                (max_x, min_y, max_z),
                (min_x, min_y, min_z),
                (min_x, min_y, max_z),
                (max_x, min_y, min_z),
                (min_x, max_y, min_z),
            ]:
                obj = self.check_raycast(corner, d)
                vertical_nexts.add(obj)

        return vertical_nexts

    @property
    def states(self) -> dict:
        return self._states

    # === internal helper ====

    def _process_cfgs(self) -> None:
        """
        Processes the configuration for the object.

        This method initializes the object's name, category, and links based on the
        configuration provided.
        Additionally, if the object has any attachments specified in the configuration,
        they are added using the `add_attachment` method.
        Finally, it adds default object states to the object
        using the `_add_states` method.
        """
        self._obj_name = self._obj_cfg.get("baselink", "rigid_obj")
        self._category = self._obj_cfg.get("category")
        self._baselink = RigidPrim(
            f"{self._path_prefix}/{self._obj_name}", self._obj_name
        )

        # if this object has vkc attachment
        if "attachments" in self._obj_cfg:
            for att in self._obj_cfg["attachments"]:
                self.add_attachment(att)

        # add default states
        self._add_states()

    def _add_states(self) -> None:
        """
        Adds default states to the object.

        This method fetches the default object states
        using `get_default_object_states` function.
        It then iterates over each state type and initializes it
        with the current object.
        The initialized states are stored in the `_states` dictionary with
        their class names as keys.
        """

        # pylint: disable=import-outside-toplevel
        from tongverse.object_states import get_default_object_states

        states = get_default_object_states()
        for state_type in states:
            self._states[state_type.__name__] = state_type(obj=self)
