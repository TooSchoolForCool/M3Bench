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


Rigdeback_franka_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(RootPath.AGENT / "Ridgeback_franka" / "main.usd"),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # base
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,
            # franka arm
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 2.999,
            "panda_joint7": 0.741,
            # tool
            "panda_finger_joint.*": 0.035,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_.*"],
            velocity_limit=10000.0,
            effort_limit=1e30,
            stiffness=1e17,
            damping=2e15,
        ),
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=1e30,
            velocity_limit=10000.0,
            stiffness=1e15,
            damping=2e13,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=1e30,
            velocity_limit=10000.0,
            stiffness=1e14,
            damping=2e12,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=2000,
            velocity_limit=10000.0,
            stiffness=1e13,
            damping=2e11,
        ),
    },
)

"""Configuration of Franka arm with Franka Hand on a Clearpath
Ridgeback base using implicit actuator models.

The following control configuration is used:

* Base: velocity control with damping
* Arm: position control with damping (contains default position offsets)
* Hand: mimic control

"""
