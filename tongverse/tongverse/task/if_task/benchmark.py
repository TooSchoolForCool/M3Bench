from __future__ import annotations

import json
from pathlib import Path

import yaml

from tongverse.task.if_task.if_env import IFTaskEnv
from tongverse.task.task_cfg import TaskCfg


class Benchmark:
    def __init__(self, task_dir: Path) -> None:
        with open(
            task_dir.joinpath("task_config.yml"),
            "r",
            encoding="UTF-8",
        ) as file:
            task_config = yaml.safe_load(file)
        self.env: IFTaskEnv = IFTaskEnv(TaskCfg.from_dict(task_config))
        self.tasks = []
        for file_path in task_dir.rglob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                self.tasks.append(json.load(f))

    def run_task(self, task, task_solver):
        info = self.env.set_task(task)
        task_solver.init(info)
        while True:
            if self.env.terminated:
                break
            action, render, require_obs = task_solver.make_decision()
            feedback = self.env.step(action, render, require_obs)
            print(feedback)

    def run(self, task_solver):
        suc_cnt = 0
        total = len(self.tasks)
        for task in self.tasks:
            print(f'Run {task["name"]} ...')
            self.run_task(task, task_solver)
            if self.env.check_task_success():
                suc_cnt += 1
        suc_rate = suc_cnt * 1.0 / total
        self.env.pause()
        return suc_rate, suc_cnt, total
