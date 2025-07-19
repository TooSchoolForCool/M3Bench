import builtins
import unittest

import numpy as np

# pylint: enable=wrong-import-position
from tongverse.sim.simulator import Simulator
from tongverse.sim.simulator_cfg import SimulatorCfg


class TestSetup(unittest.TestCase):
    def tearDown(self):
        """Stops simulator after each test."""
        # clear the stage
        Simulator.clear_instance()

    def test_singleton(self):
        """Test if the Simulator instance is a singleton."""
        sim1 = Simulator()
        sim2 = Simulator()
        self.assertIs(sim1, sim2)

        # Clear the instance and check if it's None
        sim2.clear_instance()
        self.assertTrue(sim1.instance() is None)

        # Check if a new instance is created after clearing
        sim3 = Simulator()
        self.assertIsNot(sim1, sim3)
        self.assertIs(sim1.instance(), sim3.instance())

    def test_initialization(self):
        """Test if Simulator initializes with the correct configurations."""
        cfg = SimulatorCfg()
        sim = Simulator(cfg)
        # check valid settings
        self.assertEqual(sim.get_physics_context().get_physics_dt(), cfg.sim_params.dt)
        self.assertEqual(
            sim.get_rendering_dt(), cfg.sim_params.dt * cfg.sim_params.substeps
        )
        self.assertEqual(
            sim.get_physics_context().is_ccd_enabled(), cfg.sim_params.enable_ccd
        )
        self.assertEqual(
            sim.get_physics_context().is_stablization_enabled(),
            cfg.sim_params.enable_stabilization,
        )
        # check valid gravity
        gravity_dir, gravity_mag = sim.get_physics_context().get_gravity()
        gravity = np.array(gravity_dir) * gravity_mag
        np.testing.assert_almost_equal(gravity, cfg.sim_params.gravity, decimal=6)

        # check GPU buffer setting
        self.assertEqual(
            sim.get_physics_context().is_gpu_dynamics_enabled(), cfg.sim_params.use_gpu
        )
        np.testing.assert_almost_equal(
            sim.get_physics_context().get_bounce_threshold(),
            cfg.sim_params.bounce_threshold_velocity,
        )
        np.testing.assert_almost_equal(
            sim.get_physics_context().get_friction_offset_threshold(),
            cfg.sim_params.friction_offset_threshold,
        )

    def test_launched_from_terminal(self):
        self.assertFalse(builtins.ISAAC_LAUNCHED_FROM_TERMINAL)  # pylint: disable=E1101


if __name__ == "__main__":
    unittest.main()
