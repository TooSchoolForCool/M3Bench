from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class TaskCfg:
    # TODO strict format instead of dict?
    task_type: str
    planning: dict
    agents: list[dict]

    @classmethod
    def from_dict(cls, config):
        valid_fields = {f.name for f in fields(TaskCfg)}
        key_mapping = {
            "type": "task_type",
        }
        filtered_data = {}
        for key, value in config.items():
            data_key = key_mapping.get(key, key)
            if data_key in valid_fields:
                filtered_data[data_key] = value

        return TaskCfg(**filtered_data)
