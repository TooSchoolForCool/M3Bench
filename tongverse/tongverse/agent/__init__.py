from __future__ import annotations

import copy

from tongverse.agent.default import DefaultAgent
from tongverse.agent.default import Husky_ur5_cfg as DefaultHusky_ur5_cfg
from tongverse.agent.default import Mec_kinova_cfg as DefaultMec_kinova_cfg
from tongverse.agent.default import Rigdeback_franka_cfg as DefaultRigdeback_franka_cfg
from tongverse.agent.user_defined import (
    Husky_ur5_cfg,
    Mec_kinova_cfg,
    MobileManiputor,
    Rigdeback_franka_cfg,
)

# noqa: N806
# pylint: disable= C0103
DEFAULT_AGENT_CFG = {
    "Ridgeback_franka": DefaultRigdeback_franka_cfg,
    "Mec_kinova": DefaultMec_kinova_cfg,
    "Husky_ur5": DefaultHusky_ur5_cfg,
}
CUSTOM_AGENT_CFG = {
    "Ridgeback_franka": Rigdeback_franka_cfg,
    "Mec_kinova": Mec_kinova_cfg,
    "Husky_ur5": Husky_ur5_cfg,
}
# flake8: noqa


# pylint: disable=import-outside-toplevel
def process_base_joint_spawn_position(agent, articulation_cfg, spawn_offset):
    if spawn_offset is not None:
        spawn_offset = list(spawn_offset)
        if agent == "Ridgeback_franka":
            if articulation_cfg.default_joint_state.joint_pos is None:
                articulation_cfg.default_joint_state.joint_pos = {}
                articulation_cfg.default_joint_state.joint_pos[
                    "dummy_base_prismatic_x_joint"
                ] = spawn_offset[0]
                articulation_cfg.default_joint_state.joint_pos[
                    "dummy_base_prismatic_y_joint"
                ] = spawn_offset[1]
                articulation_cfg.default_joint_state.joint_pos[
                    "dummy_base_revolute_z_joint"
                ] = spawn_offset[2]
            else:
                articulation_cfg.default_joint_state.joint_pos[
                    "dummy_base_prismatic_x_joint"
                ] += spawn_offset[0]
                articulation_cfg.default_joint_state.joint_pos[
                    "dummy_base_prismatic_y_joint"
                ] += spawn_offset[1]
                articulation_cfg.default_joint_state.joint_pos[
                    "dummy_base_revolute_z_joint"
                ] += spawn_offset[2]
        elif agent in ["Mec_kinova", "Husky_ur5"]:
            if articulation_cfg.default_joint_state.joint_pos is None:
                articulation_cfg.default_joint_state.joint_pos = {}
                articulation_cfg.default_joint_state.joint_pos[
                    "base_y_base_x"
                ] = spawn_offset[0]
                articulation_cfg.default_joint_state.joint_pos[
                    "base_theta_base_y"
                ] = spawn_offset[1]
                articulation_cfg.default_joint_state.joint_pos[
                    "base_link_base_theta"
                ] = spawn_offset[2]
            else:
                articulation_cfg.default_joint_state.joint_pos[
                    "base_y_base_x"
                ] += spawn_offset[0]
                articulation_cfg.default_joint_state.joint_pos[
                    "base_theta_base_y"
                ] += spawn_offset[1]
                articulation_cfg.default_joint_state.joint_pos[
                    "base_link_base_theta"
                ] += spawn_offset[2]

    return articulation_cfg


def get_agent(agent_cfg):
    agent = agent_cfg.get("agent", "Ridgeback_franka")
    if agent_cfg.get("user_defined_actuator_model", False) is True:
        articulation_cfg = copy.deepcopy(CUSTOM_AGENT_CFG[agent])
        # articulation_cfg = copy.deepcopy(Mec_kinova_cfg)
    else:
        articulation_cfg = copy.deepcopy(DEFAULT_AGENT_CFG[agent])

    spawn_offset = agent_cfg.get("base_joint_pos_spawn_position", None)
    # update articulation_cfg
    articulation_cfg = process_base_joint_spawn_position(
        agent, articulation_cfg, spawn_offset
    )

    if not agent_cfg.get("user_defined_actuator_model", False):
        agent = DefaultAgent(articulation_cfg, agent_cfg)
    else:
        agent = MobileManiputor(articulation_cfg, agent_cfg)

    return agent
