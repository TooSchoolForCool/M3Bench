import unittest

from tongverse.env import Env
from tongverse.env.env_base import default_env_cfg


class TestEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the environment and scene."""
        cfg = default_env_cfg
        cfg.scene["name"] = "House_23"
        cls.env = Env(cfg)
        cls.scene = cls.env.get_scene()
        cls.env.reset()

    def tearDown(self):
        """Restart the simulator after each test."""
        self.env.reset(soft=True)

    def test_reset(self):
        """Test stopping the simulation and starting again"""
        position = [-5.3, -5.6, 0.01483]
        target_obj = self.scene.get_object_by_name("remotecontrol_23_link")
        old_position = target_obj.get_world_pose()[0]

        target_obj.set_world_pose(position=position)
        self.env.step()

        real_world_position = target_obj.get_world_pose()[0]

        for i in range(3):
            self.assertAlmostEqual(real_world_position[i].item(), position[i], places=2)

        self.env.reset()

        reset_world_position = target_obj.get_world_pose()[0]
        for i in range(3):
            self.assertAlmostEqual(
                reset_world_position[i].item(), old_position[i].item(), places=2
            )

    def test_get_scene_name(self):
        name = self.env.get_scene().name
        self.assertTrue(name == "House_23")

    # def test_get_obj_state(self):
    #     state = self.env.get_env_state()
    #     print(f"\033[95mOBJ_STATE: {state}")


if __name__ == "__main__":
    unittest.main()
