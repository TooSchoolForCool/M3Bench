from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields

import torch


class IFTaskDetector(ABC):
    """
    Interface for task success condition detectors.
    """

    _registry = {}

    @property
    @abstractmethod
    def register_name(self):
        pass

    @abstractmethod
    def setup(self, env):
        pass

    @abstractmethod
    def detect(self, env) -> bool:
        pass

    @classmethod
    def from_dict(cls, config) -> "IFTaskDetector":
        cls_name = config["type"]
        target_class = cls._registry[cls_name]
        valid_fields = {f.name for f in fields(target_class)}
        filterd_config = {k: v for k, v in config.items() if k in valid_fields}

        return target_class(**filterd_config)

    @classmethod
    def register(cls, name):
        def decorator(x):
            cls._registry[name] = x
            return x

        return decorator


@dataclass
@IFTaskDetector.register(name="is_moved")
class IsMoved(IFTaskDetector):
    def __post_init__(self) -> None:
        self.pos = None

    @property
    def register_name(self):
        return "is_moved"

    def setup(self, env):
        self.pos = deepcopy(env.get_agent().get_world_poses()[0])

    def detect(self, env) -> bool:
        return not torch.equal(env.get_agent().get_world_poses()[0], self.pos)


@dataclass
@IFTaskDetector.register(name="is_on")
class IsOn(IFTaskDetector):
    target_name: str
    supported_name: str

    @property
    def register_name(self):
        return "is_on"

    def setup(self, env):
        return

    def detect(self, env) -> bool:
        target_obj = env.get_scene().get_object_by_name(self.target_name + "_link")
        supported_obj = env.get_scene().get_object_by_name(
            self.supported_name + "_link")
        is_supported = target_obj.states["SupportedBy"].get_value(supported_obj)
        return is_supported


@dataclass
@IFTaskDetector.register(name="all_on")
class AllOn(IFTaskDetector):
    target_category: str
    supported_name: str

    @property
    def register_name(self):
        return "all_on"

    def setup(self, env):
        return

    def detect(self, env) -> bool:
        for target_obj in env.get_scene().get_objects_by_category(self.target_category):
            supported_obj = env.get_scene().get_object_by_name(
                self.supported_name + "_link")
            is_supported = target_obj.states["SupportedBy"].get_value(supported_obj)
            if not is_supported:
                return is_supported
        return True


@dataclass
@IFTaskDetector.register(name="is_in")
class IsIn(IFTaskDetector):
    target_name: str
    container_name: str

    @property
    def register_name(self):
        return "is_in"

    def setup(self, env):
        return

    def detect(self, env) -> bool:
        target_obj = env.get_scene().get_object_by_name(self.target_name + "_link")
        container_obj = env.get_scene().get_object_by_name(
            self.container_name + "_link")
        return (container_obj.states["Inside"].get_value() is False) and\
               (target_obj.states["Inside"].get_value(container_obj) is True)


@dataclass
@IFTaskDetector.register(name="nav_obj")
class NavObj(IFTaskDetector):
    target_name: str
    max_distance: str

    @property
    def register_name(self):
        return "nav_obj"

    def setup(self, env):
        return

    def detect(self, env) -> bool:
        target_obj = env.get_scene().get_object_by_name(self.target_name + "_link")
        agent = env.get_agent()
        agent_x, agent_y, _ = agent.get_world_pose()[0]
        target_x, target_y, _ = target_obj.get_world_pose()[0]
        distance = ((agent_x - target_x) ** 2 + (agent_y - target_y) ** 2) ** 0.5
        return distance < float(self.max_distance)
