from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
from omni.isaac.core.utils.types import ArticulationAction, JointsState


# pylint: disable=duplicate-code
class Agent(ABC):
    @abstractmethod
    def __init__(self, articulation_cfg, agent_cfg: dict):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def apply_action(self, target_action: ArticulationAction):
        pass

    @abstractmethod
    def get_applied_action(self):
        pass

    @abstractmethod
    def set_gains(
        self,
        kps: Optional[torch.Tensor] = None,
        kds: Optional[torch.Tensor] = None,
        joint_indices: list[int] = None,
    ):
        pass

    @abstractmethod
    def get_gains(self, joint_indices: list[int] = None) -> Tuple:
        pass

    @abstractmethod
    def set_joint_state(
        self,
        positions=None,
        velocities=None,
        efforts=None,
        joint_indices: list[int] = None,
    ):
        pass

    @abstractmethod
    def get_joint_state(
        self,
        joint_indices: Optional[Sequence[int]] = None,
    ) -> JointsState:
        pass

    @abstractmethod
    def get_joints_default_state(self):
        pass

    @abstractmethod
    def set_joint_position_targets(
        self, positions: torch.Tensor, joint_indices: list[int] = None
    ):
        pass

    @abstractmethod
    def set_joint_velocity_targets(
        self, velocities: torch.Tensor, joint_indices: list[int] = None
    ):
        pass

    @abstractmethod
    def get_joint_names(self) -> Tuple[str]:
        pass

    @abstractmethod
    def get_body_names(self) -> Tuple[str]:
        pass

    @property
    @abstractmethod
    def agent_name(self):
        pass

    @property
    @abstractmethod
    def prim_path(self):
        pass
