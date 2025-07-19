from __future__ import annotations

from tongverse.utils.constant import RootPath

from .mobile_manipulator_cfg import (
    DefaultJointStateCfg,
    ImplicitActuatorCfg,
    MetaInfoCfg,
    MobileManipulatorCfg,
)

# pylint: disable=duplicate-code
Rigdeback_franka_cfg = MobileManipulatorCfg(
    meta_info=MetaInfoCfg(
        usd_path=str(RootPath.AGENT / "Ridgeback_franka" / "main.usd"),
        base_names=(
            "dummy_base_prismatic_x_joint",
            "dummy_base_prismatic_y_joint",
            "dummy_base_revolute_z_joint",
        ),
        arms_names=(
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ),
        end_effector_name="endeffector",
        grippers_names=("panda_finger_joint1", "panda_finger_joint2"),
    ),
    default_joint_state=DefaultJointStateCfg(
        joint_pos={
            "dummy_base_prismatic_x_joint": 0,
            "dummy_base_prismatic_y_joint": 0,
            "dummy_base_revolute_z_joint": 0,
            "panda_joint1": 0,
            "panda_joint2": -0.569,
            "panda_joint3": 0,
            "panda_joint4": -2.810,
            "panda_joint5": 0,
            "panda_joint6": 2.999,
            "panda_joint7": 0.741,
            "panda_finger_joint1": 0.035,
            "panda_finger_joint2": 0.035,
        }
    ),
    default_actuator_state={
        "dummy_base_prismatic_x_joint": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e17,
            damping=2e15,
        ),
        "dummy_base_prismatic_y_joint": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e17,
            damping=2e15,
        ),
        "dummy_base_revolute_z_joint": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e17,
            damping=2e15,
        ),
        "panda_joint1": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e15,
            damping=2e13,
        ),
        "panda_joint2": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e15,
            damping=2e13,
        ),
        "panda_joint3": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e15,
            damping=2e13,
        ),
        "panda_joint4": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e15,
            damping=2e13,
        ),
        "panda_joint5": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e14,
            damping=2e12,
        ),
        "panda_joint6": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e14,
            damping=2e12,
        ),
        "panda_joint7": ImplicitActuatorCfg(
            effort_limit=1e30,
            stiffness=1e14,
            damping=2e12,
        ),
        "panda_finger_joint1": ImplicitActuatorCfg(
            effort_limit=2000,
            stiffness=1e13,
            damping=2e11,
        ),
        "panda_finger_joint2": ImplicitActuatorCfg(
            effort_limit=2000,
            stiffness=1e13,
            damping=2e11,
        ),
    },
)
