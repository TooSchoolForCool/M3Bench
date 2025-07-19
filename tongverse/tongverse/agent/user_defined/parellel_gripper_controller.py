from __future__ import annotations

from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper


class ParallelGripperController(ParallelGripper):
    # pylint: disable=useless-parent-delegation
    def open(self):
        """Applies actions to the articulation that opens the gripper
        (ex: to release an object held)."""
        super().open()

    # pylint: disable=useless-parent-delegation
    def close(self):
        """Applies actions to the articulation that closes the gripper
        (ex: to hold an object)."""
        super().close()

    def forward(self, action: str) -> ArticulationAction:
        """calculates the ArticulationAction for all of the articulation
        joints that corresponds to "open"
           or "close" actions.

        Args:
            action (str): "open" or "close" as an abstract action.

        Raises:
            Exception: _description_

        Returns:
            ArticulationAction: articulation action to be passed to the articulation
            itself (includes all joints of the articulation).
        """
        if action == "open":
            if self._action_deltas is None:
                target_joint_positions = [
                    self._joint_opened_positions[0],
                    self._joint_opened_positions[1],
                ]
            else:
                current_joint_positions = self._get_joint_positions_func()
                current_left_finger_position = current_joint_positions[
                    self._joint_dof_indicies[0]
                ]
                current_right_finger_position = current_joint_positions[
                    self._joint_dof_indicies[1]
                ]
                target_joint_positions = [
                    current_left_finger_position + self._action_deltas[0],
                    current_right_finger_position + self._action_deltas[1],
                ]
        elif action == "close":
            if self._action_deltas is None:
                target_joint_positions = [
                    self._joint_closed_positions[0],
                    self._joint_closed_positions[1],
                ]
            else:
                current_joint_positions = self._get_joint_positions_func()
                current_left_finger_position = current_joint_positions[
                    self._joint_dof_indicies[0]
                ]
                current_right_finger_position = current_joint_positions[
                    self._joint_dof_indicies[1]
                ]
                target_joint_positions = [
                    current_left_finger_position - self._action_deltas[0],
                    current_right_finger_position - self._action_deltas[1],
                ]

        else:
            raise ValueError(
                f"action {action} is not defined for ParallelGripper"
            )
        return ArticulationAction(
            joint_positions=target_joint_positions,
            joint_indices=[self._joint_dof_indicies[0], self._joint_dof_indicies[1]],
        )
