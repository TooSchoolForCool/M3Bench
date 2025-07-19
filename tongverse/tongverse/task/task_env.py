from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any

import torch
from omni.isaac.core.utils.types import ArticulationAction

from tongverse.env.env_base import Env
from tongverse.env.env_cfg import default_env_cfg
from tongverse.motion_planner import get_motion_planner
from tongverse.sensor import Camera, default_camera_cfg
from tongverse.task.task_cfg import TaskCfg


class TaskAction(ABC, ArticulationAction):
    class ManipulationLevel(Enum):
        NON_ENV = 0
        ARTICULATION = 1
        DISCRETE = 2

    @property
    @abstractmethod
    def level(self) -> ManipulationLevel:
        pass

    @property
    def name(self):
        return self.__class__.name

    @abstractmethod
    def as_dict(self) -> dict:
        pass


class TaskEnv(ABC, Env):
    def __init__(self, config: TaskCfg) -> None:
        # Set env and cam
        self.env_cfg = deepcopy(default_env_cfg)
        self.env_cfg.agent["name"] = config.agents[0]["type"]
        super().__init__(self.env_cfg)
        if "camera" in config.agents[0]:
            cfg = default_camera_cfg
            cfg.cam_params["parent"] = self._agent.prim_path
            self.camera = Camera(cfg)
            self.camera.set_local_pose(
                translation=config.agents[0]["camera"]["local_pose"]
            )

        # Set planning settings
        self._terminated = True
        self._manipulation_level = config.planning["manipulation_level"]
        if "motion_planner" in config.planning:
            self._motion_planner = get_motion_planner(
                config.planning["motion_planner"], self, self._agent
            )
        self._terminated = True
        self._step_cnt = 0

    def set_task(self, data, reset=True) -> dict[str, Any]:
        """
        Update env to setup a specfic task
        """
        if not self.terminated:
            # TODO maybe a log.warn?
            raise RuntimeError("Set a new task but old task is not terminated yet.")
        self.env_cfg.scene.update(data["scene"])
        self.reload_scene(self.env_cfg.scene)
        self._terminated = False
        if reset:
            self.reset()
        return {}

    def _motion_plan(self, target_action: TaskAction) -> tuple:
        """
        Convert task-level action to env-executable articulation actions
        """
        if target_action.level == TaskAction.ManipulationLevel.ARTICULATION:
            return target_action
        assert self._motion_planner is not None
        assert target_action.name in self._motion_planner.action_space
        return self._motion_planner.generate_action(target_action.as_dict())

    def step(
        self, target_action: TaskAction, render: bool = True, require_obs: bool = False
    ) -> tuple[bool, dict]:
        assert not self.terminated
        self._step_cnt += 1
        if target_action.level != TaskAction.ManipulationLevel.NON_ENV:
            try:
                articulation = self._motion_plan(target_action)
                result = (True, {})
            except AssertionError:
                return (False, {"error": "Not supported action."})
            if target_action.level == TaskAction.ManipulationLevel.DISCRETE:
                position = (
                    torch.tensor([articulation[0]])
                    if articulation[0] is not None
                    else None
                )
                orientation = (
                    torch.tensor([articulation[1]])
                    if articulation[1] is not None
                    else None
                )
                self._agent.set_world_poses(position, orientation)
                super().step()
            else:
                raise NotImplementedError()
        else:
            result = self._step_non_env_action(target_action)
        if require_obs:
            result[1]["observation"] = self.get_observation()
        return result

    @abstractmethod
    def _step_non_env_action(self, target_action: TaskAction) -> tuple[bool, dict]:
        pass

    @property
    def terminated(self) -> bool:
        return self._terminated

    def end_task(self):
        self._terminated = True

    @abstractmethod
    def get_observation(self):
        """
        Make observation according to env_state.
        """

    @abstractmethod
    def check_task_success(self) -> bool:
        pass
