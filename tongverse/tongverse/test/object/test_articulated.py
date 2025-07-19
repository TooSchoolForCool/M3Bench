# pylint: disable=C0411
import unittest  # noqa: I001
import numpy as np
import torch
from tongverse.env.env_base import Env, default_env_cfg

# NOTE: Ensure that the tongverse module is imported
# before omni.isaac to avoid any dependency conflicts.
from omni.isaac.core.utils.rotations import euler_angles_to_quat


# pylint: skip-file
class TestArticulated(unittest.TestCase):
    """Test cases for articulated objects in the environment.
    NOTE:
    1. In this test scenario, object named storagefurniture_95_link
    does not have a fixed joint linking it to the world, whereas the
    object storagefurniture_97_link does have such a fixed joint linking it to the world
    2. For objects with fixed joints linked to the world, their velocities will remain
    zero even after calling the set_velocity method
    """

    @classmethod
    def setUpClass(cls):
        """Set up the environment and scene."""
        cfg = default_env_cfg
        cfg.scene["name"] = "physcene_1141"
        cfg.agent = None
        cls.env = Env(cfg)
        cls.env.reset()
        cls.scene = cls.env.get_scene()

    def tearDown(self):
        """Restart the simulator after each test."""
        self.env.reset(soft=True)

    def test_joint_names(self):
        """Test retrieving joint names of objects."""
        target_obj = self.scene.get_object_by_name("storagefurniture_95_link")
        self.assertTrue(
            target_obj.joint_names
            == {"storagefurniture_95_link_joint_0", "storagefurniture_95_link_joint_1"}
        )
        target_obj = self.scene.get_object_by_name("storagefurniture_97_link")
        self.assertTrue(target_obj.joint_names == {"storagefurniture_97_link_joint_1"})

    def test_position(self):
        """Test setting and getting object positions."""
        position = [1.0, 2.0, 3.0]
        target_obj = self.scene.get_object_by_name("storagefurniture_95_link")
        target_obj.set_world_pose(position=position)
        real_world_position = target_obj.get_world_pose()[0]
        for i in range(3):
            self.assertAlmostEqual(real_world_position[i].item(), position[i], places=1)

        # Test setting and getting local pose
        # target_obj.disable_physics(False)
        self.env.step()
        translation = [0.1, 0.1, 0.1]
        target_obj.set_local_pose(translation)
        real_local_translation = target_obj.get_local_pose()[0]
        for i in range(3):
            self.assertAlmostEqual(
                real_local_translation[i].item(), translation[i], places=1
            )

    def test_orientation(self):
        """Test setting and getting object orientations."""
        orientation = np.array(euler_angles_to_quat([45, -60, 180], degrees=True))
        target_obj = self.scene.get_object_by_name("storagefurniture_95_link")
        target_obj.set_world_pose(orientation=orientation)
        real_world_orientation = target_obj.get_world_pose()[1]
        for i in range(3):
            self.assertAlmostEqual(
                real_world_orientation[i].item(), orientation[i], places=2
            )
        self.env.step()

        orientation = np.array(euler_angles_to_quat([45, -60, 180], degrees=True))
        target_obj.set_local_pose(orientation=orientation)
        real_local_orientation = target_obj.get_local_pose()[1]
        for i in range(3):
            self.assertAlmostEqual(
                real_local_orientation[i].item(), orientation[i], places=2
            )

    def test_joint_position(self):
        """Test setting and getting joint positions
        on articulated object without fixed joint link to world.
        FIXME: Test fail"""
        target_obj1 = self.scene.get_object_by_name("storagefurniture_95_link")
        target_obj1._fixed = False
        target_obj1.set_joint_positions("storagefurniture_95_link_joint_0", 1.5)
        self.env.step()
        new_pos = next(
            iter(
                target_obj1.get_joint_positions(
                    "storagefurniture_95_link_joint_0"
                ).values()
            )
        )
        self.assertAlmostEqual(new_pos, 1.5, places=1)

    def test_joint_position2(self):
        """Test setting and getting joint positions
        on articulated object with fixed joint link to world."""
        target_obj2 = self.scene.get_object_by_name("storagefurniture_97_link")
        target_obj2.set_joint_positions("storagefurniture_97_link_joint_1", 0.3)
        new_pos = next(
            iter(
                target_obj2.get_joint_positions(
                    "storagefurniture_97_link_joint_1"
                ).values()
            )
        )
        self.assertAlmostEqual(new_pos, 0.3, places=1)

    def test_disable_gravity(self):
        """Test disable gravity on articulated object without
        fixed joint link to world
        FIXME: Test failed"""
        target_obj = self.scene.get_object_by_name("storagefurniture_95_link")
        target_obj._fixed = False
        real_world_pose = target_obj.get_world_pose()
        new_position = real_world_pose[0].tolist()
        # Raise the object by increasing its Z-axis position by 1.5 units.
        new_position[-1] += 1.5

        target_obj.disable_gravity(True)
        target_obj.set_world_pose(position=new_position)
        for _ in range(20):
            self.env.step()

        real_world_position = target_obj.get_world_pose()[0]

        for i in range(3):
            self.assertAlmostEqual(
                real_world_position[i].item(), new_position[i], places=2
            )
        # Release the object
        target_obj.disable_physics(False)
        for _ in range(100):
            self.env.step()
        real_world_position = target_obj.get_world_pose()[0]
        self.assertNotEqual(real_world_position[-1].item(), new_position[-1])

    def test_mass(self):
        """Test setting and getting object masses."""
        target_obj = self.scene.get_object_by_name("storagefurniture_95_link")
        mass = 5
        target_obj.set_mass(mass)
        self.env.step()
        real_mass = target_obj.get_mass()
        self.assertAlmostEqual(mass, real_mass)

        target_obj = self.scene.get_object_by_name("storagefurniture_97_link")
        mass = 5
        target_obj.set_mass(mass)
        self.env.step()
        real_mass = target_obj.get_mass()
        self.assertAlmostEqual(mass, real_mass)

    def test_density(self):
        """Test setting and getting object densities."""
        target_obj = self.scene.get_object_by_name("storagefurniture_95_link")
        density = 0.6
        target_obj.set_density(density)
        self.env.step()
        real_density = target_obj.get_density()
        self.assertAlmostEqual(density, real_density)

        target_obj = self.scene.get_object_by_name("storagefurniture_97_link")
        density = 0.6
        target_obj.set_density(density)
        self.env.step()
        real_density = target_obj.get_density()
        self.assertAlmostEqual(density, real_density)

    def test_linear_velocity(self):
        """Test setting and getting linear velocity of objects on articulated
        object without fixed joint link to world.
        FIXME: Test fail"""
        target_obj = self.scene.get_object_by_name("storagefurniture_95_link")

        # To test velocity in Z: raise the object by
        # increasing its Z-axis position by 2 units.
        real_world_pose = target_obj.get_world_pose()
        new_position = real_world_pose[0].tolist()
        new_position[-1] += 2
        target_obj.set_world_pose(position=new_position)

        linear_velocity = torch.Tensor([0.1, 0.2, 0.3])
        target_obj.keep_still()
        target_obj.set_linear_velocity(linear_velocity)
        self.env.step()
        real_linear_velocity = target_obj.get_linear_velocity()
        print(
            f"\033[95m cabinet_338_link linealar velocity: {linear_velocity} "
            f"{real_linear_velocity}"
        )
        self.assertTrue(
            np.isclose(
                linear_velocity.cpu().numpy(),
                real_linear_velocity.cpu().numpy(),
                atol=1e-02,
            ).all()
        )

    def test_linear_velocity2(self):
        """Test setting and getting linear velocity of objects on articulated
        object with fixed joint link to world"""
        # NOTE: objects with fixed joints linked to the world,
        # their velocities will remain zero even after calling the
        # set_linear_velocity method
        target_obj = self.scene.get_object_by_name("storagefurniture_97_link")
        linear_velocity = torch.Tensor([0.1, 0.2, 0.3])
        target_obj.set_linear_velocity(linear_velocity)
        self.env.step()
        real_linear_velocity = target_obj.get_linear_velocity()

        self.assertTrue(
            np.isclose(
                np.zeros(3), real_linear_velocity.cpu().numpy(), atol=1e-02
            ).all()
        )

    def test_angular_velocity(self):
        """Test setting and getting angular velocity of objects on articulated
        object without fixed joint link to world.
        FIXME: Test fail"""
        target_obj = self.scene.get_object_by_name("storagefurniture_95_link")
        angular_velocity = torch.Tensor([0.1, 0.2, 0.3])
        target_obj.keep_still()
        target_obj.set_angular_velocity(angular_velocity)
        self.env.step()
        real_angular_velocity = target_obj.get_angular_velocity()
        self.assertTrue(
            np.isclose(
                angular_velocity.cpu().numpy(),
                real_angular_velocity.cpu().numpy(),
                atol=1e-02,
            ).all()
        )

    def test_angular_velocity2(self):
        """Test setting and getting linear velocity of objects on articulated
        object with fixed joint link to world"""
        # NOTE: objects with fixed joints linked to the world,
        # their velocities will remain zero even after calling the set_velocity method
        target_obj = self.scene.get_object_by_name("storagefurniture_97_link")
        angular_velocity = torch.Tensor([0.1, 0.2, 0.3])
        target_obj.set_angular_velocity(angular_velocity)
        self.env.step()
        real_angular_velocity = target_obj.get_angular_velocity()
        print(f"\033[95m angular velocity: {angular_velocity} {real_angular_velocity}")
        self.assertTrue(
            np.isclose(
                np.zeros(3), real_angular_velocity.cpu().numpy(), atol=1e-02
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
