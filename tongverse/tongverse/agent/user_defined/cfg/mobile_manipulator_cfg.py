from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

from tongverse.sensor.perception_based_sensor import Camera


@dataclasses.dataclass
class ActuatorCfg:
    """Configuration for actuators in an articulation
    If none, use value frm usd."""

    # -- limits
    effort_limit: Optional[float] = None
    # -- gains
    stiffness: Optional[float] = None
    damping: Optional[float] = None
    # --properties
    armatures: Optional[float] = None
    frictions: Optional[float] = None


@dataclasses.dataclass
class ImplicitActuatorCfg(ActuatorCfg):
    """
    The PD control is handled implicitly by the simulation.
    In implicitActuator, all values should be set to the simulation directly.
    If none, use value frm usd
    """


@dataclasses.dataclass
class ExplicitActuatorCfg(ActuatorCfg):
    """
    The gains and limits are processed by the actuator model
    By default, set gains to zero, and torque limit to a high value in simulation
    to avoid any interference
    """

    effort_limit: float = 1.0e9
    stiffness: float = 0
    damping: float = 0


@dataclasses.dataclass
class MetaInfoCfg:
    """Meta-information about the robot."""

    usd_path: str

    base_names: Tuple[str, ...]
    """Order must match prim order in usd"""
    arms_names: Tuple[str, ...]
    """Order must match prim order in usd"""
    end_effector_name: str
    grippers_names: Optional[Tuple[str, ...]] = None
    gripper_open_position: List[float] = None
    """joint positions of the left finger joint and
    the right finger joint respectively when opened."""
    gripper_closed_position: List[float] = None
    """joint positions of the left finger joint and
    the right finger joint respectively when closed."""
    action_deltas: List[float] = None
    """deltas to apply for finger joint positions
    when openning or closing the gripper. Defaults to None."""
    perception_based_sensor: Optional[Tuple[Camera]] = None


class DefaultJointStateCfg:
    def __init__(
        self, joint_pos: Dict[str, float] = None, joint_vel: Dict[str, float] = None
    ):
        self.joint_pos = joint_pos
        """Joint positions of the joints. If None, set value to 0.0 for all joints."""
        self.joint_vel = joint_vel
        """Joint velocities of the joints. If None, set value  to 0.0 for all joints."""


class MobileManipulatorCfg:
    """Configuration parameters for a mobile manipulator."""

    def __init__(
        self,
        meta_info: MetaInfoCfg,
        default_joint_state: DefaultJointStateCfg = None,
        default_actuator_state: Dict[str, ActuatorCfg] = None,
    ):
        self.meta_info = meta_info
        """Meta-information about the robot."""
        self.default_joint_state = (
            default_joint_state
            if default_joint_state is not None
            else DefaultJointStateCfg()
        )
        """Initial state of the robot."""
        self.default_actuator_state: Dict[str, ActuatorCfg] = default_actuator_state
