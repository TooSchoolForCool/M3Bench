# pylint: disable=C0411
import unittest  # noqa: I001

import numpy as np
import torch


from tongverse.env.env_base import Env, default_env_cfg
from omni.isaac.core.utils.rotations import euler_angles_to_quat


# pylint: skip-file
class TestRigid(unittest.TestCase):
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

    def test_position(self):
        """Test setting and getting object positions."""

        position = (1.0, 2.0, 3.0)
        rigid_objects_dict = self.scene.get_all_rigid_objects()
        first_rigid_obj_name = next(iter(rigid_objects_dict))
        target_obj = self.scene.get_object_by_name(first_rigid_obj_name)
        target_obj.set_world_pose(position=position)
        real_world_position = target_obj.get_world_pose()[0]

        for i in range(3):
            self.assertAlmostEqual(real_world_position[i].item(), position[i], places=1)

        self.env.step()
        translation = (0.1, 0.1, 0.1)
        target_obj.set_local_pose(translation)
        real_local_translation = target_obj.get_local_pose()[0]

        for i in range(3):
            self.assertAlmostEqual(
                real_local_translation[i].item(), translation[i], places=1
            )

    def test_orientation(self):
        """Test setting and getting object orientations."""

        orientation = np.array(euler_angles_to_quat((45, -60, 180), degrees=True))
        target_obj = self.scene.get_object_by_name("egg_62_link")
        target_obj.set_world_pose(orientation=orientation)
        self.env.step()
        real_world_orientation = target_obj.get_world_pose()[1]
        for i in range(3):
            self.assertAlmostEqual(
                real_world_orientation[i].item(), orientation[i], places=1
            )
        self.env.step()

        orientation = np.array(euler_angles_to_quat((45, -60, 180), degrees=True))
        target_obj.set_local_pose(orientation=orientation)
        self.env.step()
        real_local_orientation = target_obj.get_local_pose()[1]
        for i in range(3):
            self.assertAlmostEqual(
                real_local_orientation[i].item(), orientation[i], places=2
            )

    def test_baselink(self):
        """Test the root link of the object with multiple visual mesh."""

        target_obj = self.scene.get_object_by_name("laptop_38_link")
        obj_name = target_obj.name
        self.assertTrue("laptop_38_link" == obj_name)

    def test_disable_physics(self):
        """Test freezing an object's pose."""

        target_obj = self.scene.get_object_by_name("egg_62_link")
        real_world_pose = target_obj.get_world_pose()
        new_position = real_world_pose[0].tolist()
        new_orient = np.array(
            euler_angles_to_quat([30, -60, 180], degrees=True)
        ).tolist()
        new_position[-1] += 2
        target_obj.set_world_pose(position=new_position, orientation=new_orient)
        target_obj.disable_physics(True)

        for _ in range(500):
            self.env.step()

        real_world_position = target_obj.get_world_pose()[0]
        for i in range(3):
            self.assertAlmostEqual(
                real_world_position[i].item(), new_position[i], places=2
            )
        real_world_orientation = target_obj.get_world_pose()[1]
        for i in range(3):
            self.assertAlmostEqual(
                real_world_orientation[i].item(), new_orient[i], places=2
            )

    def test_enable_physics(self):
        """Test unfreezing an object."""
        target_obj = self.scene.get_object_by_name("egg_62_link")
        real_world_pose = target_obj.get_world_pose()
        new_position = real_world_pose[0].tolist()
        new_position[-1] += 2
        target_obj.set_world_pose(position=new_position)
        target_obj.disable_physics(True)

        for _ in range(10):
            self.env.step()

        real_world_position = target_obj.get_world_pose()[0]
        self.assertAlmostEqual(
            real_world_position[-1].item(), new_position[-1], places=2
        )

        target_obj.disable_physics(False)
        for _ in range(10):
            self.env.step()

        real_world_position = target_obj.get_world_pose()[0]
        self.assertNotEqual(real_world_position[-1].item(), new_position[-1])

    def test_mass(self):
        """Test setting and getting the mass of an object."""

        target_obj = self.scene.get_object_by_name("egg_62_link")
        mass = 5
        target_obj.set_mass(mass)
        self.env.step()
        real_mass = target_obj.get_mass()
        self.assertAlmostEqual(mass, real_mass)

    def test_mass2(self):
        """Test setting and getting the mass of an articulated object."""

        target_obj = self.scene.get_object_by_name("laptop_38_link")
        mass = 5
        target_obj.set_mass(mass)
        self.env.step()
        real_mass = target_obj.get_mass()
        self.assertAlmostEqual(mass, real_mass)

    def test_density(self):
        """Test setting and getting the density of an object."""

        target_obj = self.scene.get_object_by_name("egg_62_link")
        density = 0.6
        target_obj.set_density(density)
        self.env.step()
        real_density = target_obj.get_density()
        self.assertAlmostEqual(density, real_density)

    def test_density2(self):
        """Test setting and getting the density of an articulated object."""

        target_obj = self.scene.get_object_by_name("laptop_38_link")
        density = 0.6
        target_obj.set_density(density)
        self.env.step()
        real_density = target_obj.get_density()
        self.assertAlmostEqual(density, real_density)

    def test_linear_velocity(self):
        """Test setting and getting the linear velocity of an object."""
        target_obj = self.scene.get_object_by_name("egg_62_link")
        linear_velocity = torch.Tensor([0.1, 0.1, 0.1])
        target_obj.set_angular_velocity(torch.zeros(3))
        target_obj.set_linear_velocity(linear_velocity)
        target_obj.disable_gravity(True)
        self.env.step()
        real_linear_velocity = target_obj.get_linear_velocity()
        self.assertTrue(
            np.isclose(
                linear_velocity.cpu().numpy(),
                real_linear_velocity.cpu().numpy(),
                atol=1e-02,
            ).all()
        )

    def test_linear_velocity2(self):
        """Test setting and getting the linear velocity of an articulated object."""
        target_obj = self.scene.get_object_by_name("laptop_38_link")
        linear_velocity = torch.Tensor([0.1, 0.2, 0.3])
        target_obj.set_angular_velocity(torch.zeros(3))
        target_obj.set_linear_velocity(linear_velocity)
        target_obj.disable_gravity(True)
        self.env.step()
        real_linear_velocity = target_obj.get_linear_velocity()
        print(f"\033[95m linealar velocity: {linear_velocity} {real_linear_velocity}")
        self.assertTrue(
            np.isclose(
                linear_velocity.cpu().numpy(),
                real_linear_velocity.cpu().numpy(),
                atol=1e-02,
            ).all()
        )

    def test_angular_velocity(self):
        """Test setting and getting the angular velocity of an object."""
        target_obj = self.scene.get_object_by_name("egg_62_link")
        angular_velocity = torch.Tensor([0.1, 0.2, 0.3])
        target_obj.set_linear_velocity(torch.zeros(3))
        target_obj.disable_gravity(True)
        self.env.step()
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
        """Test setting and getting the angular velocity of an object."""
        target_obj = self.scene.get_object_by_name("laptop_38_link")
        angular_velocity = torch.Tensor([0.1, 0.2, 0.2])
        target_obj.set_linear_velocity(torch.zeros(3))
        target_obj.disable_gravity(True)
        self.env.step()
        target_obj.set_angular_velocity(angular_velocity)
        self.env.step()
        real_angular_velocity = target_obj.get_angular_velocity()
        print(f"\033[95m Angular velocity: {angular_velocity} {real_angular_velocity}")
        self.assertTrue(
            np.isclose(
                angular_velocity.cpu().numpy(),
                real_angular_velocity.cpu().numpy(),
                atol=1e-02,
            ).all()
        )

    def test_disable_gravity(self):
        target_obj = self.scene.get_object_by_name("egg_62_link")
        real_world_pose = target_obj.get_world_pose()
        new_position = real_world_pose[0].tolist()
        new_position[-1] += 2
        target_obj.set_world_pose(position=new_position)
        target_obj.disable_gravity(True)
        for _ in range(10):
            self.env.step()

        real_world_position = target_obj.get_world_pose()[0]
        self.assertAlmostEqual(
            real_world_position[-1].item(), new_position[-1], places=2
        )

        target_obj.disable_gravity(False)
        for _ in range(10):
            self.env.step()

        real_world_position = target_obj.get_world_pose()[0]
        self.assertNotEqual(real_world_position[-1].item(), new_position[-1])


if __name__ == "__main__":
    unittest.main()
