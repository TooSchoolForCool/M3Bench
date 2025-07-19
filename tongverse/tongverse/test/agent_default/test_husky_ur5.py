# pylint: disable=duplicate-code
import unittest  # noqa: I001

import torch

from torch import tensor

# pylint: disable=import-error,wrong-import-order
from tongverse.env import Env
from tongverse.env.env_cfg import default_env_cfg
from omni.isaac.core.utils.types import ArticulationAction


class TestJoint(unittest.TestCase):
    """
    NOTE: In this test scenario, we test Ridgeback_franka with Orbit API
    """

    @classmethod
    def setUpClass(cls):
        """Set up the environment and scene."""
        cfg = default_env_cfg
        # cfg.agent[0]["agent"] = "Ridgeback_franka"
        cfg.agent["agent"] = "Husky_ur5"
        cfg.agent["user_defined_actuator_model"] = False
        cfg.scene = None

        cls.env = Env(cfg)
        cls.env.reset()
        cls.agent = cls.env.get_agent()

    def tearDown(self):
        """Restart the simulator after each test."""
        self.env.reset(soft=True)

    def test_agent_name(self):
        self.assertTrue(self.agent.agent_name == "Husky_ur5")

    def test_get_joint_names(self):
        self.assertTrue(
            self.agent.get_joint_names()
            == (
                # base
                "base_y_base_x",
                "base_theta_base_y",
                "base_link_base_theta",
                # ur arm
                "ur_arm_shoulder_pan_joint",
                "ur_arm_shoulder_lift_joint",
                "ur_arm_elbow_joint",
                "ur_arm_wrist_1_joint",
                "ur_arm_wrist_2_joint",
                "ur_arm_wrist_3_joint",
            )
        )

    def test_get_joint_id_by_name(self):
        joint_ids = [0, 3, 5]
        self.assertTrue(
            self.agent.get_joint_names().index("base_y_base_x") == joint_ids[0]
        )

        self.assertTrue(
            self.agent.get_joint_names().index("ur_arm_shoulder_pan_joint")
            == joint_ids[1]
        )  # joint_ids[1]

        self.assertTrue(
            self.agent.get_joint_names().index("ur_arm_elbow_joint") == joint_ids[2]
        )

    def test_get_name_id_by_id(self):
        self.assertTrue(self.agent.get_joint_names()[0] == "base_y_base_x")
        self.assertTrue(self.agent.get_joint_names()[3] == "ur_arm_shoulder_pan_joint")
        self.assertTrue(self.agent.get_joint_names()[5] == "ur_arm_elbow_joint")

    def test_get_set_joint_state_all(self):
        joint_pos = tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        for _ in range(2):
            self.env.step()
            self.agent.set_joint_state(joint_pos)

        current_joint_state = self.agent.get_joint_state()

        for i in range(9):
            self.assertAlmostEqual(
                current_joint_state.positions[i].item(), joint_pos[i].item(), places=2
            )

    def test_get_set_joint_state_by_joint_id(self):
        joint_pos = tensor([0.392, 0.1455, 0.633, 0.1])
        for _ in range(2):
            self.env.step()
        # pylint:disable=E1123,E1120
        self.agent.set_joint_state(positions=joint_pos, joint_indices=[0, 1, 2, 3])

        current_joint_state = self.agent.get_joint_state([0, 1, 2, 3])

        for i in range(4):
            self.assertAlmostEqual(
                current_joint_state.positions[i].item(), joint_pos[i].item(), places=2
            )

    def test_get_set_joint_pos_target_all(self):
        joint_pos_target = tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.001, 0.001])
        desired_action = ArticulationAction(joint_positions=joint_pos_target)

        self.agent.apply_action(desired_action)
        self.env.step()
        target = self.agent.get_applied_action().joint_positions
        for i in range(9):
            self.assertAlmostEqual(
                joint_pos_target[i].item(), target[i].item(), places=2
            )

    def test_get_set_joint_pos_target_by_id(self):
        joint_ids = [0, 5, 7]
        joint_pos_target = torch.Tensor([1.4, 1.7, 2.1])
        self.agent.set_joint_position_targets(joint_pos_target, joint_ids)
        self.env.step()
        target = self.agent.get_applied_action().joint_positions[joint_ids]

        for ind in range((len(joint_ids))):
            self.assertAlmostEqual(
                joint_pos_target[ind].item(), target[ind].item(), places=2
            )

    def test_get_set_joint_vel_target_all(self):
        joint_vel_target = tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.001, 0.001])
        applied_data = ArticulationAction(joint_velocities=joint_vel_target)

        self.agent.apply_action(applied_data)
        self.env.step()
        target = self.agent.get_applied_action().joint_velocities

        for i in range(9):
            self.assertAlmostEqual(
                joint_vel_target[i].item(), target[i].item(), places=2
            )

    def test_get_set_joint_vel_target_by_id(self):
        joint_ids = [0, 5, 7]
        joint_vel_target = torch.Tensor([0.1, 0.1, 0.1])

        self.agent.set_joint_velocity_targets(joint_vel_target, joint_ids)
        self.env.step()

        target = self.agent.get_applied_action().joint_velocities[joint_ids]

        for ind in range(len(joint_ids)):
            self.assertAlmostEqual(
                joint_vel_target[ind].item(), target[ind].item(), places=2
            )

    def test_get_set_joint_effort_target_all(self):
        joint_effort_target = tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.001, 0.001])
        applied_data = ArticulationAction(joint_efforts=joint_effort_target)
        self.agent.apply_action(applied_data)
        self.env.step()
        target_from_buffer = self.agent.get_joint_effort_target()
        target_from_phyx = self.agent.root_physx_view.get_dof_actuation_forces()[0, :]
        for i in range(9):
            self.assertAlmostEqual(
                joint_effort_target[i].item(), target_from_buffer[i].item(), places=2
            )
        for i in range(9):
            self.assertAlmostEqual(
                joint_effort_target[i].item(), target_from_phyx[i].item(), places=2
            )

    def test_get_set_joint_effort_target_by_id(self):
        joint_ids = [0, 5, 7]
        joint_effort_target = torch.Tensor([0.1, 0.4, 0.7])
        applied_data = ArticulationAction(
            joint_efforts=joint_effort_target, joint_indices=joint_ids
        )

        self.agent.apply_action(applied_data)
        self.env.step()
        target_from_buffer = self.agent.get_joint_effort_target()
        target_from_phyx = self.agent.root_physx_view.get_dof_actuation_forces()[0, :]
        for ind, val in enumerate(joint_ids):
            self.assertAlmostEqual(
                joint_effort_target[ind].item(),
                target_from_buffer[val].item(),
                places=2,
            )
            self.assertAlmostEqual(
                joint_effort_target[ind].item(), target_from_phyx[val].item(), places=2
            )

    def test_reset(self):
        joint_pos_target = tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.001, 0.001])
        joint_vel_target = tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        applied_data = ArticulationAction(
            joint_positions=joint_pos_target, joint_velocities=joint_vel_target
        )
        self.agent.apply_action(applied_data)
        self.env.step()
        self.agent.reset()
        current_joint_state = self.agent.get_joint_state()
        default_joint_pos = self.agent.get_joints_default_state().positions

        default_joint_vel = self.agent.get_joints_default_state().velocities
        for i in range(9):
            self.assertAlmostEqual(
                current_joint_state.positions[i].item(),
                default_joint_pos[i].item(),
                places=2,
            )
            self.assertAlmostEqual(
                current_joint_state.velocities[i].item(),
                default_joint_vel[i].item(),
                places=2,
            )
        # pylint:disable = W0212
        spawn_offset = default_env_cfg.agent["base_joint_pos_spawn_position"]
        current_position = self.agent.get_joint_state([0, 1, 2]).positions

        for i in range(3):
            self.assertAlmostEqual(
                spawn_offset[i],
                current_position[i].item(),
                places=2,
            )

    def test_gains(self):
        stiffness = tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.001, 0.001])
        damping = tensor([5, 0.5, 0.5, 0.5, 5, 0.5, 5, 0.001, 0.001])
        self.agent.set_gains(stiffness, damping)
        self.env.step()
        cur_stiffness, cur_damping = self.agent.get_gains()
        for i in range(9):
            self.assertAlmostEqual(
                cur_stiffness[i].item(), stiffness[i].item(), places=2
            )
            self.assertAlmostEqual(cur_damping[i].item(), damping[i].item(), places=2)

    def test_gains_by_id(self):
        joint_ids = [0, 5, 7]
        stiffness = tensor([0.5, 0.5, 0.5])
        damping = tensor([5, 0.5, 5])
        self.agent.set_gains(stiffness, damping, joint_ids)
        self.env.step()
        cur_stiffness, cur_damping = self.agent.get_gains(joint_ids)
        for i in range(3):
            self.assertAlmostEqual(
                cur_stiffness[i].item(), stiffness[i].item(), places=2
            )
            self.assertAlmostEqual(cur_damping[i].item(), damping[i].item(), places=2)


if __name__ == "__main__":
    unittest.main()
