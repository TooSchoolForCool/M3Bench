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


Mec_kinova_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        activate_contact_sensors=False,
        usd_path=str(RootPath.AGENT / "Mec_kinova" / "main.usd"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # base joints
            # x joint
            "base_y_base_x": 0,
            # y joint
            "base_theta_base_y": 0,
            # theta joint
            "base_link_base_theta": 0.0,
            # arm joints
            "joint_1": 0,
            "joint_2": 0,
            "joint_3": 0,
            "joint_4": 0,
            "joint_5": 0,
            "joint_6": 0,
            "joint_7": 0,
            "end_hand_prismatic_joint_left": 0.0,
            "end_hand_prismatic_joint_right": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            velocity_limit=10000.0,
            joint_names_expr=["base_.*"],
            effort_limit=1e30,
            stiffness=6e19,
            damping=6e17,
        ),
        "arm1": ImplicitActuatorCfg(
            joint_names_expr=["joint_1", "joint_2"],
            effort_limit=1e29,
            velocity_limit=9000.0,
            stiffness=7e19,
            damping=6e17,
        ),
        "arm3": ImplicitActuatorCfg(
            joint_names_expr=["joint_3", "joint_4"],
            velocity_limit=8000.0,
            effort_limit=1e29,
            stiffness=2e19,
            damping=5e17,
        ),
        "arm4": ImplicitActuatorCfg(
            joint_names_expr=["joint_5"],
            effort_limit=1e29,
            stiffness=6e16,
            velocity_limit=7000.0,
            damping=6e14,
        ),
        "arm5": ImplicitActuatorCfg(
            joint_names_expr=["joint_6", "joint_7"],
            effort_limit=1e30,
            velocity_limit=10000.0,
            stiffness=9e12,
            damping=9e10,
        ),
        "arm6": ImplicitActuatorCfg(
            joint_names_expr=[
                "end_hand_prismatic_joint_left",
                "end_hand_prismatic_joint_right",
            ],
            effort_limit=1e30,
            velocity_limit=10000.0,
            stiffness=9e10,
            damping=9e8,
        ),
    },
)
"""Configuration of Franka arm with Franka Hand on a
Clearpath Ridgeback base using implicit actuator models.

The following control configuration is used:

* Base: velocity control with damping
* Arm: position control with damping (contains default position offsets)
* Hand: mimic control

"""
