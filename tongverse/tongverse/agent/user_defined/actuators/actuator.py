from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from omni.isaac.core.utils.types import ArticulationAction

from .registry import actuator_registry


class ActuatorBase(ABC):
    def __init__(self, agent):
        self.agent = agent
        self.computed_effort = None
        self.applied_effort = None

    @abstractmethod
    def compute_effort(
        self,
        action: ArticulationAction,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ):
        raise NotImplementedError

    def _clip_effort(
        self, action: ArticulationAction, effort: torch.Tensor
    ) -> torch.Tensor:
        """Clip the desired torques based on the motor limits.

        Args:
            desired_torques: The desired torques to clip.

        Returns:
            The clipped torques.
        """
        if action.joint_indices is None:
            return torch.clip(
                effort,
                min=-self.agent.default_effort_limit,
                max=self.agent.default_effort_limit,
            )

        return torch.clip(
            effort,
            min=-self.agent.default_effort_limit[action.joint_indices],
            max=self.agent.default_effort_limit[action.joint_indices],
        )


@actuator_registry.register_model(name="ImplicitActuator")
class ImplicitActuator(ActuatorBase):
    def compute_effort(
        self,
        action: ArticulationAction,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ):
        # compute errors
        joint_num = (
            self.agent.joint_num
            if action.joint_indices is None
            else len(action.joint_indices)
        )

        positions = (
            torch.zeros(joint_num)
            if action.joint_positions is None
            else action.joint_positions
        )
        velocities = (
            torch.zeros(joint_num)
            if action.joint_velocities is None
            else action.joint_velocities
        )
        efforts = (
            torch.zeros(joint_num)
            if action.joint_efforts is None
            else action.joint_efforts
        )

        error_pos = positions - joint_pos
        error_vel = velocities - joint_vel
        # calculate the desired joint torques
        stiffness, dampings = self.agent.get_gains(joint_indices=action.joint_indices)
        self.computed_effort = stiffness * error_pos + dampings * error_vel + efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(action, self.computed_effort)
        return action


@actuator_registry.register_model(name="IdealPDActuator")
class IdealPDActuator(ActuatorBase):
    def compute_effort(
        self,
        action: ArticulationAction,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ):
        # compute errors
        joint_num = (
            self.agent.joint_num
            if action.joint_indices is None
            else len(action.joint_indices)
        )
        positions = (
            torch.zeros(joint_num)
            if action.joint_positions is None
            else action.joint_positions
        )
        velocities = (
            torch.zeros(joint_num)
            if action.joint_velocities is None
            else action.joint_velocities
        )
        efforts = (
            torch.zeros(joint_num)
            if action.joint_efforts is None
            else action.joint_efforts
        )
        error_pos = positions - joint_pos
        error_vel = velocities - joint_vel
        # calculate the desired joint torques
        stiffness, dampings = self.agent.get_gains(joint_indices=action.joint_indices)
        self.computed_effort = stiffness * error_pos + dampings * error_vel + efforts
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(action, self.computed_effort)
        action.joint_positions = None
        action.joint_velocities = None
        action.joint_efforts = self.applied_effort
        return action
