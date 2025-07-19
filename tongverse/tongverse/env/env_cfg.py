from __future__ import annotations

import dataclasses
from typing import Any, Optional, Tuple

from tongverse.sim.simulator_cfg import default_simulator_cfg


class EnvCfg:
    """
    Configuration class for the environment.
    """

    @dataclasses.dataclass
    class SceneCfg:
        """
        Configuration for a scene within the environment.
        """

        name: str = "physcene_1141"
        position: Optional[Tuple[float, float, float]] = (0, 0, 0)
        orientation: Optional[Tuple[float, float, float, float]] = (1, 0, 0, 0)
        obj_semantic_label: str = "baselink"
        """
        Semantic label for objects. Options: 'baselink' or 'category'.
        """

    @dataclasses.dataclass
    class AgentCfg:
        """
        Configuration for an agent within the environment.
        """

        agent: str = "Ridgeback_franka"
        """
        Supported agents: 'Ridgeback_franka', 'Mec_kinova', 'Husky_ur5', 'Bruce'.
        """
        position: Optional[Tuple[float, float, float]] = (0, 0, 0)
        orientation: Optional[Tuple[float, float, float, float]] = (1, 0, 0, 0)
        base_joint_pos_spawn_position: Optional[Tuple] = (0, 0, 0)
        """
        The spawn_offset tuple represents the joint position offset of followings:
        ("dummy_base_prismatic_x_joint",
        "dummy_base_prismatic_y_joint",
        "dummy_base_revolute_z_joint")
        By default, the dummy base offset is 0.
        """
        user_defined_actuator_model: bool = True
        """
        Indicates whether a user-defined actuator model should be used.
        If set to False, the actuator model from Orbit will be used by default.
        """

    @dataclasses.dataclass
    class ViewCfg:
        """
        Configuration for the view within the environment.
        """

        light_mode: str = "camera"
        """
        Mode for lighting: 'stage' or 'camera'.
        """

    @dataclasses.dataclass
    class GroundCfg:
        """
        Configuration for the ground within the environment.
        """

        z_position: float = 0
        name: str = "ground_plane"
        """Prim name of the ground"""
        prim_path: str = "/World/groundPlane"
        """Prim path of the ground"""
        visible: bool = False
        """Visibility of the ground."""

    def __init__(self):
        self.scene: dict[str, Any] = dataclasses.asdict(self.SceneCfg())
        """We support multi-scene added to stage"""
        self.agent: dict[str, Any] = dataclasses.asdict(self.AgentCfg())
        """We support multi-scene added to stage"""
        self.view_cfg = dataclasses.asdict(self.ViewCfg())
        self.ground_cfg = dataclasses.asdict(self.GroundCfg())
        self.simulator_cfg = default_simulator_cfg


default_env_cfg = EnvCfg()
