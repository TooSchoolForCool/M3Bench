from __future__ import annotations

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.stage import get_current_stage
from pxr import PhysxSchema

# pylint: enable=wrong-import-position
from tongverse.sim.simulator_cfg import SimulatorCfg, default_simulator_cfg


class Simulator(SimulationContext):
    """Simulator is singleton object representing the simulation context."""

    def __init__(
        self,
        cfg: SimulatorCfg = default_simulator_cfg,
        use_gpu_dynamic: bool = False,
        gpu_found_lost_aggregate_pairs_capacity: int = 2**25,
    ):
        """
        Parameters:
            cfg (SimulatorCfg, optional): The configuration for the environment.
            Defaults to the configuration from EnvCfg.
            use_gpu_dynamic (bool): Enable/disable GPU accelerated dynamics simulation.
                Default is False
            gpu_found_lost_aggregate_pairs_capacity (int): Capacity of found and lost \
            buffers in aggregate system allocated in GPU global memory.
            Default is 2**25.


        FIXME:
            Currently, GPU pipeline usage (with Carb) doesn't work as expected,
            and we encountered the following errors:

            1. Segmentation Fault Error occurs when using carb.settings.get_settings().
            set_bool("/physics/suppressReadback", True).

            2. Runtime Error: Expected all tensors to be on the same device but found at
            least two devices, cpu and cuda:0!.
            More details can be found at:
            https://forums.developer.nvidia.com/t/how-can-i-use-different-graphics-cards-for-different-rl-tasks/239010.

            3. When calling the dynamic control API to get_dof_state(), we get a PhysX
            error: PxArticulationReducedCoordinate::copyInternalStateToCache:
            it is illegal to call this method if PxSceneFlag::eENABLE_DIRECT_GPU_API is
            enabled!

            Hack:
            @Qi As a workaround, we use the CPU as the device but enable \
            GPU-accelerated dynamics simulation using PhysxSceneAPI directly \
            without Carb.


        NOTE:
            1. If GPU-accelerated dynamics simulation is enabled, it might be \
                necessary to increase the "Gpu Found Lost Aggregate Pairs Capacity" \
                value, The value depends on the physics scene.

            2. Also, if trying GPU pipeline usage, sim_params might be needed.

        ??? abstract "Examples"
            ``` python title=""
            def __init__(self):
                self.cfg = SimulatorCfg if cfg is None else cfg
                physic_sim_parms = dataclasses.asdict(self.cfg.sim_params)
                super().__init__(
                    physics_dt=self.cfg.sim_params.dt,
                    rendering_dt=self.cfg.sim_params.dt * self.cfg.sim_params.substeps,
                    sim_params=physic_sim_parms,
                    backend=self.cfg.backend,
                    device=self.cfg.device,
                    set_defaults=False,
                )

            ```
        """

        self.cfg = cfg
        super().__init__(
            physics_dt=self.cfg.sim_params["dt"],
            rendering_dt=self.cfg.sim_params["dt"] * self.cfg.sim_params["substeps"],
            backend=self.cfg.backend,
            device=self.cfg.device,
            set_defaults=True,
        )
        if use_gpu_dynamic:
            stage = get_current_stage()
            physics_scene = stage.GetPrimAtPath("/physicsScene")
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene)
            physx_scene.CreateEnableGPUDynamicsAttr().Set(True)
            physx_scene.CreateBroadphaseTypeAttr().Set("GPU")
            physx_scene.GetGpuFoundLostAggregatePairsCapacityAttr().Set(
                gpu_found_lost_aggregate_pairs_capacity
            )

    def step(self, render: bool = True) -> None:
        """
        Perform a simulation step.

        Args:
            render (bool): Whether to render the simulation.
        """
        if self.is_stopped():
            # if stop, exit the app
            self.shutdown()
        elif not self.is_playing():
            # if pause, keep UI alive
            self.render()
        else:
            super().step(render)

    def reset(self, soft: bool = False) -> None:
        """
        Reset the simulator.

        Args:
            soft (bool): If True, set objects state back to default state and keep \
            origin simulation view, otherwise, stop simulator, reset simulation view,
            restart simulator, restart timeline and render.
        """

        if not soft:
            if not self.is_stopped():
                super().stop()
            # Create the physics simulation view and play the timeline
            # Step physics, render the app
            super().initialize_physics()

        elif super().physics_sim_view is None:
            self.app.print_and_log(
                "Physics simulation view is not set. Please ensure the first "
                "reset(soft=False) call is with soft=False."
            )

    def shutdown(self) -> None:
        """Stops the physics simulation and shutdown the app"""
        super().stop()
        # Shutdown app
        self.app.print_and_log("TongVerse is stopped. Shutting down the app.")
        self.app.shutdown()
