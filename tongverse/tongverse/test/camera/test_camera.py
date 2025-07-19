# pylint: disable=C0411
import unittest  # noqa: I001
from tongverse.env.env_base import Env, default_env_cfg
from tongverse.sensor import Camera, default_camera_cfg


# pylint: skip-file
class TestCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the environment and scene."""
        env_cfg = default_env_cfg
        env_cfg.scene = None
        cls.env = Env(env_cfg)
        cls.env.reset()
        agent = cls.env.get_agent()
        agent_baselink = f"{agent.prim_path}/base_link"
        cam_cfg = default_camera_cfg
        cam_cfg.cam_params["parent"] = agent_baselink
        cam_cfg.cam_params["name"] = "test_cam"
        cls.cam = Camera(cam_cfg)

    def tearDown(self):
        """Restart the simulator after each test."""
        self.env.reset(soft=True)

    def test_world_pose(self):
        pos = [1, 0, 1]
        self.cam.set_world_pose(
            position=pos,
        )
        self.env.step()
        real_pos = self.cam.get_world_pose()[0]
        for i in range(3):
            self.assertAlmostEqual(real_pos[i].item(), pos[i], places=1)

    def test_local_pose(self):
        translate = [1, 0, 1]
        orient = [1, 0, 0, 1]
        self.cam.set_local_pose(translation=translate, orientation=orient)
        self.env.step()
        real_translate = self.cam.get_local_pose()[0]
        real_orient = self.cam.get_local_pose()[1]
        for i in range(3):
            self.assertAlmostEqual(real_translate[i].item(), translate[i], places=1)
            self.assertAlmostEqual(real_orient[i].item(), orient[i], places=1)

    def test_prim_path(self):
        agent = self.env.get_agent()
        agent_baselink = f"{agent.prim_path}/base_link"
        self.assertTrue(self.cam.prim_path == f"{agent_baselink}/test_cam_Xform")

    def test_persp_cam(self):
        cfg = default_camera_cfg
        agent = self.env.get_agent()
        # NOTE: when setting perspective camera, rotation should be None
        cfg.cam_params["name"] = "persp_cam"
        cfg.cam_params["look_at"] = agent.prim_path
        cam2 = Camera(cfg)
        pos1, quat1 = cam2.get_local_pose()
        self.env.step()
        pos2, quat2 = cam2.get_local_pose()
        self.assertAlmostEqual(pos1.tolist(), pos2.tolist())
        self.assertNotEqual(quat1.tolist(), quat2.tolist())


if __name__ == "__main__":
    unittest.main()
