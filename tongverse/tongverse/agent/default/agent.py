from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.types import ArticulationAction, JointsState
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.articulation import ArticulationCfg

from tongverse.agent.base_agent import Agent


class DefaultAgent(Articulation, Agent):
    """
    NOTE: Since ORBIT only support the ``torch <https://pytorch.org/>``_ backend for
    simulation, if choose this framework, all the data structures used in the simulation
    are ``torch.Tensor`` objects.
    Quaternion orientation `(w, x, y, z)`
    In orbit, stiffness, damping in actuator can't be udapted once initiated
    """

    def __init__(self, articulation_cfg: ArticulationCfg, agent_cfg: dict):
        """
        use orbit actuator model (implicit and explicit)
        base_joint_pos_spawn_offset = {"joint1":1, "joint2":0, "joint3":3}
        # Must use "cfg.replace()" method instead of "cgf.prim_path",
        otherwise it might change the default cfg prim path
        """

        self._agent_name = None
        self._prim_path = None
        self._base_joint_pos_spawn_offset = None
        self._load = False
        self._joint_id_to_name = {}
        self._sensors = []
        self._bodies = {}  # key:body name, value: RigidPrim
        cfg = self._process_cfgs(articulation_cfg=articulation_cfg, agent_cfg=agent_cfg)
        # pylint:disable=E1120
        super().__init__(cfg)

    def set_joint_state(
        self,
        positions=None,
        velocities=None,
        efforts=None,
        joint_indices: list[int] = None,
    ):
        # Write joint positions and velocities (set dof position, velocities)
        # to the simulation
        # can't call write_joint_state_to_sim() method directly
        # since it requires pos and vel can't be None
        env_ids = slice(None)

        joint_indices = slice(joint_indices) if joint_indices is None else joint_indices

        # set into internal buffers
        if positions is not None:
            super().data.joint_pos[env_ids, joint_indices] = positions
            super().data.joint_acc[env_ids, joint_indices] = 0.0
            # set into simulation
            super().root_physx_view.set_dof_positions(
                super().data.joint_pos, indices=self._ALL_INDICES
            )
        if velocities is not None:
            super().data.joint_vel[env_ids, joint_indices] = velocities
            super()._previous_joint_vel[env_ids, joint_indices] = velocities
            super().data.joint_acc[env_ids, joint_indices] = 0.0
            # set into simulation
            super().root_physx_view.set_dof_velocities(
                super().data.joint_vel, indices=self._ALL_INDICES
            )

    def get_joints_default_state(self):
        default_pos = super().data.default_joint_pos[0]
        default_vel = super().data.default_joint_vel[0]
        return JointsState(default_pos, default_vel, None)

    def get_joint_state(
        self,
        joint_indices: Optional[Sequence[int]] = None,
    ) -> JointsState:
        """
        data in buffer. which is joint state in prev step.
        thus, to get the latest joint state, this function should be called after
        env.step()
        we set default env_id = 0 (which is used in Orbit API)
        """
        if joint_indices is None:
            joint_indices = slice(None)
        positions = super().data.joint_pos[0, joint_indices]
        # Joint positions of all joints.
        velocities = super().data.joint_vel[0, joint_indices]
        # Joint velocities of all joints.
        return JointsState(positions, velocities, None)

    def set_gains(
        self,
        kps: Optional[torch.Tensor] = None,
        kds: Optional[torch.Tensor] = None,
        joint_indices: list[int] = None,
    ):
        if joint_indices is None:
            joint_indices = slice(None)
        if kps is not None:
            # Write joint stiffness into the simulation.(set dof stiffness)
            super().write_joint_stiffness_to_sim(kps, joint_indices)

        if kds is not None:
            # Write joint damping into the simulation.(set dof damping)
            super().write_joint_damping_to_sim(kds, joint_indices)

    def get_gains(self, joint_indices: list[int] = None) -> Tuple:
        if joint_indices is None:
            joint_indices = slice(None)
        damping = super().data.joint_damping[0, joint_indices]
        # Joint damping provided to simulation.
        stiffness = super().data.joint_stiffness[0, joint_indices]
        # Joint stiffness provided to simulation.

        return stiffness, damping

    def get_joint_pos_target(self, joint_indices: Optional[Sequence[int]] = None):
        """
        Returns processed target actions from actuator models
        Retrieved from buffer. call env.step() to update buffere first
        """
        if joint_indices is None:
            joint_indices = slice(None)
        return self._joint_pos_target_sim[0, joint_indices]

    def get_joint_vel_target(self, joint_indices: Optional[Sequence[int]] = None):
        """
        Returns processed target actions from actuator models
        Retrieved from buffer. call env.step() to update buffere first
        """
        if joint_indices is None:
            joint_indices = slice(None)
        return self._joint_vel_target_sim[0, joint_indices]

    def get_joint_effort_target(self, joint_indices: Optional[Sequence[int]] = None):
        """
        Returns processed target actions from actuator models
        Retrieved from buffer. call env.step() to update buffere first
        """
        if joint_indices is None:
            joint_indices = slice(None)
        return self._joint_effort_target_sim[0, joint_indices]

    def apply_action(self, target_action: ArticulationAction = None):
        """
        ArticulationAction
        """
        if target_action is None:
            return
        # -- set joint command into buffer
        if target_action.joint_positions is not None:
            super().set_joint_position_target(
                target_action.joint_positions, target_action.joint_indices
            )

        if target_action.joint_velocities is not None:
            super().set_joint_velocity_target(
                target_action.joint_velocities, target_action.joint_indices
            )

        if target_action.joint_efforts is not None:
            super().set_joint_effort_target(
                target_action.joint_efforts, target_action.joint_indices
            )

        # -- compute joint command from buffer and apply data to simulation
        super().write_data_to_sim()

    def set_joint_position_targets(
        self, positions: torch.Tensor, joint_indices: list[int] = None
    ):
        super().set_joint_position_target(target=positions, joint_ids=joint_indices)
        super().write_data_to_sim()

    def set_joint_velocity_targets(
        self, velocities: torch.Tensor, joint_indices: list[int] = None
    ):
        super().set_joint_velocity_target(target=velocities, joint_ids=joint_indices)
        super().write_data_to_sim()

    def get_applied_action(self) -> ArticulationAction:
        positions = self.get_joint_pos_target()
        velocities = self.get_joint_vel_target()
        efforts = self.get_joint_effort_target()
        return ArticulationAction(
            joint_positions=positions,
            joint_velocities=velocities,
            joint_efforts=efforts,
        )

    def reset(self):
        # This function can only be called when sim is playing
        if not self._load:
            self._load = True

        joint_pos, joint_vel = (
            self.data.default_joint_pos.clone(),
            self.data.default_joint_vel.clone(),
        )

        # reset dof state
        super().write_joint_state_to_sim(joint_pos, joint_vel)
        super().write_root_pose_to_sim(self.data.default_root_state[:, :7].clone())
        super().write_root_velocity_to_sim(self.data.default_root_state[:, 7:].clone())

        # reset articulation internal buffers
        # pylint:disable=E1121
        super().reset(0)
        for body_name in super().body_names:
            body = RigidPrim(prim_path=f"{self._prim_path}/{body_name}", name=body_name)
            body.initialize()
            self._bodies[body_name] = body

    def get_observation(self):
        # TODO: if there's sensor on robot
        # Grabs all observations from the robot.
        # This is keyword-mapped based on each observation modality
        # "obs_modalities": ["scan", "rgb", "depth"],
        pass

    def get_joint_names(self) -> Tuple[str]:
        return tuple(self.joint_names)

    def get_body_names(self) -> Tuple[str]:
        return tuple(self.body_names)

    @property
    def agent_name(self):
        return self._agent_name

    @property
    def prim_path(self):
        return self._prim_path

    @property
    def bodies(self):
        return self._bodies

    # == Internal Helper ==
    def _process_cfgs(self, **kwargs):
        self._agent_name = f"{kwargs['agent_cfg'].get('agent')}"
        self._prim_path = f"/World/Agent/{self._agent_name}"
        cfg = kwargs["articulation_cfg"].replace(prim_path=self._prim_path)
        return cfg
