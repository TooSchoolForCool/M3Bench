from __future__ import annotations

import json
import os
from typing import Any

import torch

from tongverse.task.if_task.task_detector import IFTaskDetector
from tongverse.task.task_cfg import TaskCfg
from tongverse.task.task_env import TaskAction, TaskEnv


class IFTaskEnv(TaskEnv):
    """
    Base class for instruction-following tasks.
    """

    def __init__(self, config: TaskCfg) -> None:
        super().__init__(config)

        # Init planning cosntraints
        self._success_condition: list[IFTaskDetector] = []
        self._max_step = config.planning.get("max_step", 10)

    def _load_conditions(self, cond_data):
        self._success_condition.clear()
        for cond in cond_data:
            det = IFTaskDetector.from_dict(cond)
            self._success_condition.append(det)
            det.setup(self)

    def set_task(self, data, reset=True) -> dict[str, Any]:
        init_info = super().set_task(data, reset=False)
        if reset:
            self.reset()
        ag_data = data["agent"]
        self._agent.set_world_poses(
            positions=torch.tensor([ag_data["position"]], dtype=torch.float),
            orientations=torch.tensor([ag_data["orientation"]], dtype=torch.float),
        )
        self.sim.step()
        self._agent.update(self.sim.get_physics_dt())
        self._load_conditions(data["success_condition"])
        init_info["observation"] = self.get_observation()
        return init_info

    def _step_non_env_action(self, target_action: TaskAction) -> tuple[bool, dict]:
        if target_action.name == "stop":
            self.end_task()
        return (True, {})

    def step(
        self, target_action: TaskAction, render: bool = True, require_obs: bool = False
    ) -> tuple[bool, dict]:
        result = super().step(target_action, render, require_obs)
        result[1]["#step"] = self._step_cnt
        if self._step_cnt >= self._max_step:
            self._terminated = True
        result[1]["current_pos"] = self._agent.get_world_poses()
        return result

    def reset(self, soft: bool = False) -> None:
        """Resets this task in the environment"""
        self._step_cnt = 0
        super().reset(soft)

    def check_task_success(self):
        return all(det.detect(self) for det in self._success_condition)

    def get_observation(self, dir_name="./sensor_output/Perception_Cam/camera"):
        room_name = "这里"
        seg_json_name = _find_largest_numeric_filename(dir_name)
        print(seg_json_name)
        with open(os.path.join(dir_name, seg_json_name), "rb") as cam_file:
            dic = json.load(cam_file)
        observation = "你在{room_name}. 你看见了"
        for _, value in dic["idToLabels"].items():
            v = value["class"]
            if "room" in v:
                observation += "一个房间"
            elif (
                v != "BACKGROUND"
                and v != "UNLABELLED"
                and not v.startswith("wall")
                and "window" not in v
                and "painting" not in v
            ):
                observation += f"一个{v}, "
        observation = observation.rstrip().rstrip(",")
        if observation.endswith("你看见了"):
            observation = f"你在{room_name}. 你什么都没看见。"
        return observation


def _find_largest_numeric_filename(folder_path):
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return None

    max_numeric_filename = None
    max_number = float("-inf")

    for filename in os.listdir(folder_path):
        if "json" not in filename or "semantic_segmentation" not in filename:
            continue
        if filename[0].isdigit():
            numeric_part = int(filename.split("-")[0])
            if numeric_part > max_number:
                max_number = numeric_part
                max_numeric_filename = filename

    return max_numeric_filename
