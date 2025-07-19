# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Ridgeback-Manipulation robots.

The following configurations are available:

* :obj:`RIDGEBACK_FRANKA_PANDA_CFG`: Clearpath Ridgeback base with Franka Emika arm

Reference: https://github.com/ridgeback/ridgeback_manipulation
"""

from __future__ import annotations

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg

from tongverse.utils.constant import RootPath

# from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR


Husky_ur5_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(RootPath.AGENT / "Husky_ur5" / "main.usd"),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # base
            "base_y_base_x": 0.00001,
            ##
            "base_theta_base_y": 0.00001,
            ##
            "base_link_base_theta": 0.0001,
            # ur arm
            "ur_arm_shoulder_pan_joint": 0,
            "ur_arm_shoulder_lift_joint": 0,
            "ur_arm_elbow_joint": 0,
            "ur_arm_wrist_1_joint": 0,
            "ur_arm_wrist_2_joint": 0,
            "ur_arm_wrist_3_joint": 0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_.*"],
            velocity_limit=10000.0,
            effort_limit=1e30,
            stiffness=6e19,
            damping=6e17,
        ),
        "arm1": ImplicitActuatorCfg(
            joint_names_expr=[
                "ur_arm_shoulder_pan_joint",
                "ur_arm_shoulder_lift_joint",
            ],
            effort_limit=1e30,
            velocity_limit=10000.0,
            stiffness=6e19,
            damping=6e17,
        ),
        "arm3": ImplicitActuatorCfg(
            joint_names_expr=["ur_arm_elbow_joint"],
            effort_limit=1e30,
            velocity_limit=10000.0,
            stiffness=1e19,
            damping=5e17,
        ),
        "arm4": ImplicitActuatorCfg(
            joint_names_expr=["ur_arm_wrist_1_joint"],
            effort_limit=1e30,
            velocity_limit=10000.0,
            stiffness=6e16,
            damping=6e14,
        ),
        "arm5": ImplicitActuatorCfg(
            joint_names_expr=["ur_arm_wrist_2_joint", "ur_arm_wrist_3_joint"],
            effort_limit=1e30,
            velocity_limit=10000.0,
            stiffness=9e12,
            damping=9e10,
        ),
    },
)
"""Configuration of Franka arm with Franka Hand on
a Clearpath Ridgeback base using implicit actuator models.

The following control configuration is used:

* Base: velocity control with damping
* Arm: position control with damping (contains default position offsets)
* Hand: mimic control

"""
