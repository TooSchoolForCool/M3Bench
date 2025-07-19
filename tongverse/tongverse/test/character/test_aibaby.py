# pylint: disable=wrong-import-order
import unittest  # noqa: I001
import random

import numpy as np

from tongverse.env import Env
from tongverse.env.env_cfg import default_env_cfg
from tongverse.character import get_baby_config, Character

from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix
from pxr import Gf, Vt


class TestAIBaby(unittest.TestCase):
    """
    NOTE: In this test scenario, we test AIBaby with TongVerse API
    """

    @classmethod
    def setUpClass(cls) -> None:
        cfg = default_env_cfg
        cfg.agent = None

        aibaby_cfg = get_baby_config()
        aibaby_cfg.position = (-2.3, -4.3, 0)

        cls.env = Env(cfg)
        cls.env.reset()

        cls.character: Character = Character(aibaby_cfg)
        cls.character.load()

    def tearDown(self) -> None:
        self.env.reset(soft=True)

    def test_character_name(self):
        self.assertTrue(self.character.name == "AIBaby")

    def test_get_joint_names(self):
        self.assertTrue(
            self.character.get_joint_names()
            == [
                "Root",
                "pelvis",
                "spine_01",
                "spine_02",
                "spine_03",
                "clavicle_l",
                "upperarm_l",
                "lowerarm_l",
                "hand_l",
                "index_01_l",
                "index_02_l",
                "index_03_l",
                "middle_01_l",
                "middle_02_l",
                "middle_03_l",
                "pinky_01_l",
                "pinky_02_l",
                "pinky_03_l",
                "ring_01_l",
                "ring_02_l",
                "ring_03_l",
                "thumb_01_l",
                "thumb_02_l",
                "thumb_03_l",
                "neck_01",
                "head",
                "eye_jiont_r",
                "eyetip_jiont_r",
                "eye_jiont_l",
                "eyetip_jiont_l",
                "ffd1Lattice",
                "ffd1Lattice1",
                "ffd1Base",
                "ffd1Base1",
                "jaw_joint",
                "jawtip_joint",
                "eyelid_l",
                "eyelid_l2",
                "eyelid_r",
                "eyelid_r2",
                "eyelidDOWN_l",
                "eyelidDOWN_l2",
                "eyelidDOWN_r",
                "eyelidDOWN_r2",
                "clavicle_r",
                "upperarm_r",
                "lowerarm_r",
                "hand_r",
                "index_01_r",
                "index_02_r",
                "index_03_r",
                "middle_01_r",
                "middle_02_r",
                "middle_03_r",
                "pinky_01_r",
                "pinky_02_r",
                "pinky_03_r",
                "ring_01_r",
                "ring_02_r",
                "ring_03_r",
                "thumb_01_r",
                "thumb_02_r",
                "thumb_03_r",
                "thigh_r",
                "calf_r",
                "foot_r",
                "ball_r",
                "thigh_l",
                "calf_l",
                "foot_l",
                "ball_l",
            ]
        )

    def test_get_set_character_transform(self):
        position = np.random.rand(3) * 5
        self.character.set_world_transform(pos=position)
        self.env.step()
        self.assertTrue(
            np.allclose(self.character.get_world_transform()[0], position, atol=1e-2)
        )

        euler = np.random.rand(3) * 2 * np.pi
        quaternion = euler_angles_to_quat(euler)
        self.character.set_world_transform(rot=quaternion)
        self.env.step()
        get_rot = self.character.get_world_transform()[1]
        results = quat_to_rot_matrix(quaternion).T.dot(
            quat_to_rot_matrix(get_rot)
        ) - np.identity(3)
        self.assertTrue(np.allclose(results, np.zeros_like(results), atol=1e-2))

        pos = np.random.rand(3) * 5
        rot = euler_angles_to_quat(np.random.rand(3) * 2 * np.pi)
        self.character.set_world_transform(pos, rot)
        self.env.step()
        self.assertTrue(
            np.allclose(self.character.get_world_transform()[0], pos, atol=1e-2)
        )
        get_rot = self.character.get_world_transform()[1]
        results = quat_to_rot_matrix(rot).T.dot(
            quat_to_rot_matrix(get_rot)
        ) - np.identity(3)
        self.assertTrue(np.allclose(results, np.zeros_like(results), atol=1e-2))

    def test_get_set_joint_local_orientations(self):
        njoint = len(self.character.get_joint_names())
        joint_local_euler_angles = np.random.rand(njoint, 3) * 2 * np.pi
        joint_local_orientations = np.array(
            [euler_angles_to_quat(euler) for euler in joint_local_euler_angles]
        )
        self.character.set_joint_local_orientations(joint_local_orientations)
        self.env.step()
        results = np.array(
            [
                quat_to_rot_matrix(set_q).T.dot(quat_to_rot_matrix(get_q))
                - np.identity(3)
                for set_q, get_q in zip(
                    joint_local_orientations,
                    self.character.get_joint_local_orientations(),
                    strict=False,
                )
            ]
        )
        self.assertTrue(np.allclose(results, np.zeros_like(results), atol=1e-2))

    def test_get_set_joint_local_orientations_by_names(self):
        njoint = len(self.character.get_joint_names())
        names = random.sample(
            self.character.get_joint_names(), k=random.randint(1, njoint)
        )
        joint_euler_by_names = np.random.rand(len(names), 3) * 2 * np.pi
        joint_orientations_by_names = np.array(
            [euler_angles_to_quat(euler) for euler in joint_euler_by_names]
        )
        self.character.set_joint_local_orientations(joint_orientations_by_names, names)
        self.env.step()
        results = np.array(
            [
                quat_to_rot_matrix(set_q).T.dot(quat_to_rot_matrix(get_q))
                - np.identity(3)
                for set_q, get_q in zip(
                    joint_orientations_by_names,
                    self.character.get_joint_local_orientations(names),
                    strict=False,
                )
            ]
        )
        self.assertTrue(np.allclose(results, np.zeros_like(results), atol=1e-2))

    def test_get_set_joint_local_transforms(self):
        njoint = len(self.character.get_joint_names())
        xforms = []
        for _ in range(njoint):
            xform = Gf.Matrix4d()
            xform.SetTranslate(Gf.Vec3d(*np.random.rand(3) * 5))
            euler = np.random.rand(3) * 2 * np.pi
            quat = euler_angles_to_quat(euler)
            rotation = Gf.Rotation().SetQuat(Gf.Quatd(*quat))
            xform.SetRotate(rotation)
            xforms.append(xform)
        xforms = Vt.Matrix4dArray(xforms)
        self.character.set_joint_local_transforms(xforms)
        self.env.step()
        transforms = self.character.get_joint_local_transforms()
        self.assertEqual(len(transforms), njoint)
        results = np.array(transforms - xforms)
        self.assertTrue(
            np.allclose(results, np.zeros_like(results), atol=1e-2, equal_nan=True)
        )

    def test_get_set_joint_local_transforms_by_names(self):
        njoint = len(self.character.get_joint_names())
        names = random.sample(
            self.character.get_joint_names(), k=random.randint(1, njoint)
        )
        xforms = []
        for _ in range(len(names)):
            xform = Gf.Matrix4d()
            xform.SetTranslate(Gf.Vec3d(*np.random.rand(3) * 5))
            euler = np.random.rand(3) * 2 * np.pi
            quat = euler_angles_to_quat(euler)
            rotation = Gf.Rotation().SetQuat(Gf.Quatd(*quat))
            xform.SetRotate(rotation)
            xforms.append(xform)
        xforms = Vt.Matrix4dArray(xforms)
        self.character.set_joint_local_transforms(xforms, names)
        self.env.step()
        transforms = self.character.get_joint_local_transforms(names)
        self.assertEqual(len(transforms), len(names))
        results = np.array(transforms - xforms)
        self.assertTrue(
            np.allclose(results, np.zeros_like(results), atol=1e-2, equal_nan=True)
        )

    def test_get_set_joint_relative_orientations(self):
        njoint = len(self.character.get_joint_names())
        joint_relative_euler_angles = np.random.rand(njoint, 3) * 2 * np.pi
        joint_relative_orientations = np.array(
            [euler_angles_to_quat(euler) for euler in joint_relative_euler_angles]
        )
        self.character.set_joint_relative_orientations(joint_relative_orientations)
        self.env.step()
        results = np.array(
            [
                quat_to_rot_matrix(set_q).T.dot(quat_to_rot_matrix(get_q))
                - np.identity(3)
                for set_q, get_q in zip(
                    joint_relative_orientations,
                    self.character.get_joint_relative_orientations(),
                    strict=False,
                )
            ]
        )
        self.assertTrue(np.allclose(results, np.zeros_like(results), atol=3e-2))

    def test_get_set_joint_relative_orientations_by_names(self):
        njoint = len(self.character.get_joint_names())
        names = random.sample(
            self.character.get_joint_names(), k=random.randint(1, njoint)
        )
        joint_euler_by_names = np.random.rand(len(names), 3) * 2 * np.pi
        joint_orientations_by_names = np.array(
            [euler_angles_to_quat(euler) for euler in joint_euler_by_names]
        )
        self.character.set_joint_relative_orientations(
            joint_orientations_by_names, names
        )
        self.env.step()
        results = np.array(
            [
                quat_to_rot_matrix(set_q).T.dot(quat_to_rot_matrix(get_q))
                - np.identity(3)
                for set_q, get_q in zip(
                    joint_orientations_by_names,
                    self.character.get_joint_relative_orientations(names),
                    strict=False,
                )
            ]
        )
        self.assertTrue(np.allclose(results, np.zeros_like(results), atol=3e-2))


if __name__ == "__main__":
    unittest.main()
