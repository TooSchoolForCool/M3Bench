from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import delete_prim, get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.types import ArticulationAction, JointsState

from tongverse.agent.base_agent import Agent
from tongverse.utils import deletable_prim_check

from .parellel_gripper_controller import ParallelGripperController


class MobileManiputor(Articulation, Agent):
    def __init__(self, articulation_cfg, agent_cfg: dict) -> None:
        self._sensors = {}
        self._bodies = {}  # key:body name, value: RigidPrim
        self._base_names = ()
        self._arms_names = ()
        self._articulation_cfg = articulation_cfg
        self._default_joint_pos = None
        self._default_joint_vel = None
        self._default_joint_effort_limit = None
        self._default_joint_stiffness = None
        self._gripper_controller = None
        self._end_effector = None
        self._default_joint_damping = None
        self._default_joint_armatures = None
        self._default_joint_frictions = None

        self._agent_name = f"{agent_cfg.get('agent','agent')}"
        self._prim_path = f"/World/Agent/{self._agent_name}"
        prim = get_prim_at_path(self._prim_path)
        usd_path = articulation_cfg.meta_info.usd_path
        if not prim.IsValid():
            add_reference_to_stage(usd_path=usd_path, prim_path=self._prim_path)
        self._process_cfgs()
        # pylint:disable=E1123,E1120
        super().__init__(
            prim_path=self._prim_path,
            name=self._agent_name,
            position=agent_cfg.get("position"),
            orientation=agent_cfg.get("orientation"),
            articulation_controller=None,
        )

    def initialize(self, physics_sim_view=None) -> None:
        """Initialize physics.
        Before initializing agent physics, ensure that the simulator physics has been
        initialized
        (by calling env.reset()). Failure to follow this order may result in a
        RuntimeError.
        """
        super().initialize(physics_sim_view)
        for body_name in self._articulation_view.body_names:
            body = RigidPrim(prim_path=f"{self._prim_path}/{body_name}", name=body_name)
            body.initialize()
            self._bodies[body_name] = body

        if self._gripper_controller is not None:
            self._gripper_controller.initialize(
                physics_sim_view=physics_sim_view,
                articulation_apply_action_func=self.apply_action,
                get_joint_positions_func=self.get_joint_positions,
                set_joint_positions_func=self.set_joint_state,
                dof_names=super().dof_names,
            )

    def remove(self):
        if deletable_prim_check(self.prim_path):
            delete_prim(self.prim_path)

    # pylint:disable=W0221
    def reset(self):
        """reset.
        Before reset, ensure that the simulator physics has been
        initialized
        (by calling env.reset()). Failure to follow this order may result in a
        RuntimeError.
        """
        self.initialize()
        # -- process default state cfgs
        self._process_default_state()
        # -- reset to default gains
        self.set_gains(
            kps=self._default_joint_stiffness, kds=self._default_joint_damping
        )
        # -- set max efforts
        self.articulatiuon_controller.set_max_efforts(
            self._default_joint_effort_limit.tolist()
        )
        # -- reset to default state
        super().set_joints_default_state(
            positions=self._default_joint_pos,
            velocities=self._default_joint_vel,
        )
        super().set_linear_velocity(torch.Tensor([[0] * 3]))
        super().set_angular_velocity(torch.Tensor([[0] * 3]))
        super().post_reset()
        if self._gripper_controller is not None:
            self._gripper_controller.post_reset()
            self.articulatiuon_controller.switch_dof_control_mode(
                dof_index=self.gripper_controller.joint_dof_indicies[0], mode="position"
            )
            self.articulatiuon_controller.switch_dof_control_mode(
                dof_index=self.gripper_controller.joint_dof_indicies[1], mode="position"
            )

    def set_world_poses(
        self,
        positions: Optional[torch.Tensor] = None,
        orientations: Optional[torch.Tensor] = None,
    ):
        """Set the poses of the prims in the view with respect to the world's frame.
        quaternion is scalar-first (w, x, y, z)"""
        self._articulation_view.set_world_poses(positions, orientations)

    def get_world_poses(self):
        """Get the poses of the prims in the view with respect to the world's frame.
        quaternion is scalar-first (w, x, y, z)"""
        position, orientation = self._articulation_view.get_world_poses()
        return position[0], orientation[0]

    def get_applied_action(self) -> Optional[ArticulationAction]:
        """Get the last applied action (target positions, velocities, efforts)

        Returns:
            ArticulationAction: last applied action.
            Note: a dictionary is used as the object's string representation

        Example:

        .. code-block:: python

            >>> # last applied action: joint_positions -> [0.0, -1.0, 0.0, -2.2, 0.0,
            2.4, 0.8, 0.04, 0.04]
            >>> prim.get_applied_action()
            {'joint_positions': [0.0, -1.0, 0.0, -2.200000047683716, 0.0,
                2.4000000953674316, 0.800000011920929, 0.03999999910593033,
                0.03999999910593033],
             'joint_velocities': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             'joint_efforts': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        """
        return self.articulatiuon_controller.get_applied_action()

    def apply_action(self, target_action: ArticulationAction = None):
        """
        Args:
            target_action (ArticulationAction): actions to be applied for next
            physics step.
            should apply to all joints
        """
        if target_action is None:
            return

        if target_action.joint_efforts is not None:
            self.physics_view.set_dof_actuation_forces(
                target_action.joint_efforts, torch.tensor([0])
            )
            # set efforts to zeros to avoid interference
            target_action.joint_efforts = None
        self.articulatiuon_controller.apply_action(target_action)

    def set_max_efforts(
        self,
        values: Optional[Union[torch.tensor, list]],
        joint_indices: Optional[Union[torch.tensor, list]] = None,
    ):
        self.articulatiuon_controller.set_max_efforts(values, joint_indices)

    def get_max_efforts(self) -> torch.Tensor:
        return self.articulatiuon_controller.get_max_efforts()

    def set_gains(
        self,
        kps: Optional[torch.Tensor] = None,
        kds: Optional[torch.Tensor] = None,
        joint_indices: list[int] = None,
    ):
        """
        High stiffness makes the joints snap faster and harder to the desired target,
        and higher damping smoothes but also slows down the joint's movement to target

        * For position control, set relatively high stiffness and low damping
        (to reduce vibrations)
        * For velocity control, stiffness must be set to zero with a non-zero damping
        * For effort control, stiffness and damping must be set to zero
        """
        if joint_indices is None:
            self.articulatiuon_controller.set_gains(kps=kps, kds=kds)
            return
        self._articulation_view.set_gains(
            kps=kps, kds=kds, joint_indices=torch.tensor(joint_indices)
        )

    def get_gains(self, joint_indices: list[int] = None) -> Tuple:
        """Returns: kps, kds"""
        kps, kds = self.articulatiuon_controller.get_gains()
        if joint_indices is None:
            return kps, kds
        return kps[joint_indices], kds[joint_indices]

    def get_joint_limits(self) -> Tuple:
        return self.articulatiuon_controller.get_joint_limits()

    def get_joint_names(self) -> Tuple[str]:
        """
        Note: in isaac sim ArticulationViews API, joints include fixed joint,
        while dof do not include fixed joint
        """
        return tuple(self._articulation_view.dof_names)

    def get_body_names(self) -> Tuple[str]:
        return tuple(self._articulation_view.body_names)

    def get_joint_state(self, joint_indices: list[int] = None) -> JointsState:
        # ArticulationView returns current joint positions and velocities only
        joint_state = super().get_joints_state()
        if joint_state is None:
            return None
        if joint_indices is None:
            return joint_state
        positions = joint_state.positions[joint_indices]
        velocities = joint_state.velocities[joint_indices]
        return JointsState(positions, velocities, efforts=None)

    def get_joint_positions(self, joint_indices: list[int] = None) -> torch.Tensor:
        return self.get_joint_state(joint_indices).positions

    def set_joint_state(
        self,
        positions: Optional[torch.FloatTensor] = None,
        velocities: Optional[torch.FloatTensor] = None,
        efforts: Optional[torch.FloatTensor] = None,
        joint_indices: list[int] = None,
    ) -> None:
        """set the articulation kinematic state"""

        joint_indices = (
            torch.tensor(joint_indices) if joint_indices is not None else None
        )
        if positions is not None:
            super().set_joint_positions(positions, joint_indices)

        if velocities is not None:
            super().set_joint_velocities(velocities, joint_indices)

        if efforts is not None:
            super().set_joint_efforts(efforts, joint_indices)

    def get_joints_default_state(self):
        # pylint:disable = useless-parent-delegation
        return super().get_joints_default_state()

    def set_joint_position_targets(
        self, positions: torch.Tensor, joint_indices: list[int] = None
    ):
        self._articulation_view.set_joint_position_targets(
            positions=positions, joint_indices=joint_indices
        )

    def set_joint_velocity_targets(
        self, velocities: torch.Tensor, joint_indices: list[int] = None
    ):
        self._articulation_view.set_joint_velocity_targets(
            velocities=velocities, joint_indices=joint_indices
        )

    def update(self, dt):
        pass

    @property
    def agent_name(self):
        return self._agent_name

    @property
    def prim_path(self):
        return self._prim_path

    @property
    def default_effort_limit(self):
        """for actuator model"""
        return self._default_joint_effort_limit

    @property
    def bodies(self):
        return self._bodies

    @property
    def joint_num(self):
        """Number of DOF of the robots"""
        return super().num_dof

    @property
    def articulatiuon_controller(self):
        """a Proportional-Derivative controller that can apply position targets,
        velocity targets and efforts"""
        # check isaac.sim.core Robot-> Articulation API
        return super().get_articulation_controller()

    @property
    def base_names(self):
        return self._base_names

    @property
    def arms_names(self):
        return self._arms_names

    @property
    def gripper_controller(self) -> ParallelGripperController:
        return self._gripper_controller

    @property
    def end_effector(self) -> RigidPrim:
        return self._bodies[self._end_effector_name]

    @property
    def physics_view(self):
        # pylint:disable = W0212
        return self._articulation_view._physics_view

    # ===================== Internal helper ============================
    def _process_cfgs(self):
        """Post processing of configuration parameters."""
        # -- process meta info
        meta_info = self._articulation_cfg.meta_info
        self._sensors = (
            meta_info.perception_based_sensor
            if meta_info.perception_based_sensor is not None
            else ()
        )
        self._base_names = meta_info.base_names
        self._arms_names = meta_info.arms_names
        self._end_effector_name = meta_info.end_effector_name

        grippers_names = meta_info.grippers_names

        if grippers_names is not None:
            if meta_info.gripper_open_position is None:
                gripper_open_position = torch.tensor([0.05, 0.05]) / get_stage_units()
            else:
                gripper_open_position = torch.tensor(meta_info.gripper_open_position)
            if meta_info.gripper_closed_position is None:
                gripper_closed_position = torch.tensor([0.0, 0.0])
            else:
                gripper_closed_position = torch.tensor(
                    meta_info.gripper_closed_position
                )

            action_deltas = meta_info.action_deltas
            self._gripper_controller = ParallelGripperController(
                end_effector_prim_path=f"{self._prim_path}/{self._end_effector_name}",
                joint_prim_names=grippers_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=action_deltas,
            )

    def _process_default_state(self):
        # STEP1 -- process default joint state
        self._process_default_joint_state()
        # STEP2: By default set value from USD joint prim,
        # Update value from config if any, otherwise use the value from usd
        self._process_default_actuator_state()

    def _process_default_joint_state(self):
        # STEP1 -- process default joint state
        default_joint_state = self._articulation_cfg.default_joint_state
        joint_names = self.get_joint_names()

        self._default_joint_pos = torch.zeros(len(joint_names))
        if default_joint_state.joint_pos is not None:
            for idx, joint in enumerate(joint_names):
                self._default_joint_pos[idx] = default_joint_state.joint_pos.get(
                    joint, 0
                )
            if self._gripper_controller is not None:
                self._gripper_controller.set_default_state(
                    torch.tensor(
                        [
                            self._default_joint_pos[
                                self._gripper_controller.joint_dof_indicies[0]
                            ],
                            self._default_joint_pos[
                                self._gripper_controller.joint_dof_indicies[1]
                            ],
                        ]
                    )
                )
        self._default_joint_vel = torch.zeros(len(joint_names))
        if default_joint_state.joint_vel is not None:
            for idx, joint in enumerate(joint_names):
                self._default_joint_vel[idx] = default_joint_state.joint_vel.get(
                    joint, 0
                )

    def _process_default_actuator_state(self):
        # when use explicit actuator, by default we set gains to zero and torque limit
        # to infinite.
        self._default_joint_effort_limit = (
            self.physics_view.get_dof_max_forces().clone()[0]
        )
        self._default_joint_stiffness = self.physics_view.get_dof_stiffnesses().clone()[
            0
        ]
        self._default_joint_damping = self.physics_view.get_dof_dampings().clone()[0]
        self._default_joint_armatures = self.physics_view.get_dof_armatures().clone()[0]
        self._default_joint_frictions = (
            self.physics_view.get_dof_friction_coefficients().clone()[0]
        )
        joint_names = self.get_joint_names()

        default_actuator_state = self._articulation_cfg.default_actuator_state
        if default_actuator_state is not None:
            for idx, joint_name in enumerate(joint_names):
                if default_actuator_state.get(joint_name, None) is not None:
                    if default_actuator_state[joint_name].effort_limit is not None:
                        self._default_joint_effort_limit[idx] = default_actuator_state[
                            joint_name
                        ].effort_limit
                    if default_actuator_state[joint_name].stiffness is not None:
                        self._default_joint_stiffness[idx] = default_actuator_state[
                            joint_name
                        ].stiffness
                    if default_actuator_state[joint_name].damping is not None:
                        self._default_joint_damping[idx] = default_actuator_state[
                            joint_name
                        ].damping
                    if default_actuator_state[joint_name].armatures is not None:
                        self._default_joint_armatures[idx] = default_actuator_state[
                            joint_name
                        ].armatures
                    if default_actuator_state[joint_name].frictions is not None:
                        self._default_joint_frictions[idx] = default_actuator_state[
                            joint_name
                        ].frictions