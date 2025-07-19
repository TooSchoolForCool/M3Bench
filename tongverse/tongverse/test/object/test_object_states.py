# pylint: skip-file
import unittest  # noqa: I001

# pylint: disable=cyclic-import
from tongverse.env.env_base import Env, default_env_cfg


class TestObjectStates(unittest.TestCase):
    """
    NOTE: In this test scenario, collision shape of bowl_63_link is convex decomposition
    """

    @classmethod
    def setUpClass(cls):
        """Set up the environment and scene."""
        cfg = default_env_cfg
        cfg.scene["name"] = "physcene_1141"
        cfg.agent["agent"] = "Mec_kinova"
        cfg.agent["user_defined_actuator_model"] = True
        cls.env = Env(cfg)
        cls.env.reset()
        cls.scene = cls.env.get_scene()
        cls.agent = cls.env.get_agent()

    def tearDown(self):
        """Restart the simulator after each test."""
        self.env.reset(soft=True)

    def test_touching(self):
        """Test setting and getting object positions."""

        target_obj = self.scene.get_object_by_name("plate_63_link")

        self.assertTrue(target_obj.states["Touching"].get_value() is True)

    def test_supported_by(self):
        target_obj = self.scene.get_object_by_name("plate_63_link")
        table = self.scene.get_object_by_name("diningtable_47_link")
        self.assertTrue(target_obj.states["SupportedBy"].get_value(table) is True)

    def test_inside(self):
        tomato = self.scene.get_object_by_name("tomato_59_link")
        tomato.set_world_pose(
            position=(-10.555091911029063, -5.487283860443768, 0.8502758145332336)
        )
        self.env.step()
        bowl = self.scene.get_object_by_name("plate_63_link")

        self.assertTrue(bowl.states["Inside"].get_value() is False)
        self.assertTrue(tomato.states["Inside"].get_value(bowl) is True)

    def test_attach_to_object(self):
        tomato = self.scene.get_object_by_name("tomato_59_link")
        bowl = self.scene.get_object_by_name("plate_63_link")
        # before attach
        if_attached = tomato.states["AttachedTo"].get_value()
        self.assertTrue(if_attached is False)
        # attach tomato to bowl
        success = tomato.states["AttachedTo"].set_value(bowl)
        self.assertTrue(success is True)
        origin_position = tomato.get_world_pose()[0]
        bowl.set_world_pose(position=[2, 1, 0])
        self.env.step()

        new_position = tomato.get_world_pose()[0]
        self.assertNotEqual(new_position.tolist(), origin_position.tolist())
        if_attached = tomato.states["AttachedTo"].get_value()
        self.assertTrue(if_attached is True)
        if_attached = tomato.states["AttachedTo"].get_value(bowl)
        self.assertTrue(if_attached is True)
        self.env.reset()
        if_attached = tomato.states["AttachedTo"].get_value(bowl)
        self.assertTrue(if_attached is False)

    def test_attach_to_agent(self):
        left_end_effector = self.agent.bodies["end_effector_link"]
        tomato = self.scene.get_object_by_name("tomato_59_link")
        # before attach
        if_attached = tomato.states["AttachedTo"].get_value()
        self.assertTrue(if_attached is False)
        # attach tomato to agent
        success = tomato.states["AttachedTo"].set_value(left_end_effector)
        self.assertTrue(success is True)
        self.env.step()
        if_attached = tomato.states["AttachedTo"].get_value(left_end_effector)
        self.assertTrue(if_attached is True)
        self.env.reset()
        if_attached = tomato.states["AttachedTo"].get_value(left_end_effector)
        self.assertTrue(if_attached is False)


if __name__ == "__main__":
    unittest.main()
