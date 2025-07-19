from __future__ import annotations

# pylint: disable=E0401
from tongverse.utils.constant import RootPath

from .mobile_manipulator_cfg import (
    ImplicitActuatorCfg,
    MetaInfoCfg,
    MobileManipulatorCfg,
)

# pylint: disable=duplicate-code
Husky_ur5_cfg = MobileManipulatorCfg(
    meta_info=MetaInfoCfg(
        usd_path=str(RootPath.AGENT / "Husky_ur5" / "main.usd"),
        base_names=("base_y_base_x", "base_theta_base_y", "base_link_base_theta"),
        arms_names=(
            "ur_arm_shoulder_pan_joint",
            "ur_arm_shoulder_lift_joint",
            "ur_arm_elbow_joint",
            "ur_arm_wrist_1_joint",
            "ur_arm_wrist_2_joint",
            "ur_arm_wrist_3_joint",
        ),
        end_effector_name="ur_arm_ee_link",
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
        "ur_arm_shoulder_pan_joint": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=6e19,
            damping=6e17,
        ),
        "ur_arm_shoulder_lift_joint": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=6e19,
            damping=6e17,
        ),
        "ur_arm_elbow_joint": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e19,
            damping=5e17,
        ),
        "ur_arm_wrist_1_joint": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=6e19,
            damping=6e14,
        ),
        "ur_arm_wrist_2_joint": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=9e12,
            damping=9e10,
        ),
        "ur_arm_wrist_3_joint": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=9e12,
            damping=9e10,
        ),
    },
)
