from __future__ import annotations

from tongverse.utils.constant import RootPath

from .mobile_manipulator_cfg import (
    ImplicitActuatorCfg,
    MetaInfoCfg,
    MobileManipulatorCfg,
)

# pylint: disable=duplicate-code
Mec_kinova_cfg = MobileManipulatorCfg(
    meta_info=MetaInfoCfg(
        usd_path=str(RootPath.AGENT / "Mec_kinova" / "main.usd"),
        base_names=(
            "base_y_base_x",
            "base_theta_base_y",
            "base_link_base_theta",
        ),
        arms_names=(
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ),
        end_effector_name="end_effector_link",
        grippers_names=(
            "end_hand_prismatic_joint_left",
            "end_hand_prismatic_joint_right",
        ),
        gripper_open_position=[0, 0],
        gripper_closed_position=[-0.0425, -0.0425],
    ),
    default_actuator_state={
        "base_y_base_x": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=6e19,
            damping=6e17,
        ),
        "base_theta_base_y": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=6e19,
            damping=6e17,
        ),
        "base_link_base_theta": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=6e19,
            damping=6e17,
        ),
        "joint_1": ImplicitActuatorCfg(
            effort_limit=1e29,
            stiffness=7e19,
            damping=6e17,
        ),
        "joint_2": ImplicitActuatorCfg(
            effort_limit=1e29,
            stiffness=7e19,
            damping=6e17,
        ),
        "joint_3": ImplicitActuatorCfg(
            effort_limit=1e29,
            stiffness=2e19,
            damping=5e17,
        ),
        "joint_4": ImplicitActuatorCfg(
            effort_limit=1e29,
            stiffness=2e19,
            damping=5e17,
        ),
        "joint_5": ImplicitActuatorCfg(
            effort_limit=1e29,
            stiffness=6e12,
            damping=6e14,
        ),
        "joint_6": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=9e12,
            damping=9e10,
        ),
        "joint_7": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=9e12,
            damping=9e10,
        ),
        "end_hand_prismatic_joint_left": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=6000,
            damping=1000,
        ),
        "end_hand_prismatic_joint_right": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=6000,
            damping=1000,
        ),
    },
)
