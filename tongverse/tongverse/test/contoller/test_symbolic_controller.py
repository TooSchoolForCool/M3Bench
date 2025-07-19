# pylint: disable=C0411,E0401
import unittest  # noqa: I001
import numpy as np
from tongverse.env import Env
from tongverse.motion_planner import SymbolicPlanner
from tongverse.env.env_base import default_env_cfg
from omni.isaac.core.objects import DynamicCuboid  # noqa: C0411


def set_agent_pose(agent, target_pose):
    if target_pose[0] and target_pose[1]:
        agent.set_world_pose(target_pose[0], target_pose[0])
    elif target_pose[0]:
        agent.set_world_pose(target_pose[0])
    elif target_pose[1] is not None:
        agent.set_world_pose(orientation=target_pose[1])
    else:
        return


class TestSymbolicController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the environment and scene and the cube."""
        cfg = default_env_cfg
        cfg.scene["name"] = "physcene_1141"
        cfg.agent = None
        cls.env = Env(cfg)
        cls.scene = cls.env.get_scene()
        for ceiling in cls.env.get_scene().get_objects_by_category("ceiling"):
            ceiling.set_visibility(False)
        cls.cube = DynamicCuboid(
            prim_path="/World/Agent/cube",
            name="cube",
            position=np.array([-6.0, -3.7, 1.0]),
            scale=np.array([0.1, 0.1, 0.1]),
            size=1.0,
            color=np.array([255, 0, 0]),
        )
        cls.symbolic_planner = SymbolicPlanner(cls.env, cls.cube)
        cls.env.reset()

    def tearDown(self):
        """Restart the simulator after each test."""
        self.env.reset(soft=True)

    def test_forward(self):
        """Restart the simulator after each test."""
        self.env.step()
        before_trans = self.cube.get_world_pose()[0]
        applied_action = self.symbolic_planner.generate_action(
            {"action_type": "step_forward"})
        set_agent_pose(self.cube, applied_action)
        self.env.step()
        after_trans = self.cube.get_world_pose()[0]
        self.assertNotEqual(before_trans[0], after_trans[0])

    def test_pick(self):
        self.env.step()
        before_trans = (
            self.env.get_scene()
            .get_object_by_name("dogbed_91_link")
            .get_world_pose()[0][1]
        )
        self.symbolic_planner.generate_action(
            {"action_type": "pick", "object_name": "dogbed_91_link"}
        )
        self.env.step()
        after_trans = (
            self.env.get_scene()
            .get_object_by_name("dogbed_91_link")
            .get_world_pose()[0][1]
        )
        self.assertNotEqual(before_trans, after_trans)

    def test_turn(self):
        self.env.step()
        before_orient = self.cube.get_world_pose()[1][0]
        applied_action = self.symbolic_planner.generate_action(
            {"action_type": "turn_left"})
        set_agent_pose(self.cube, applied_action)
        self.env.step()
        after_orient = self.cube.get_world_pose()[1][0]
        self.assertNotEqual(before_orient, after_orient)


if __name__ == "__main__":
    unittest.main()
