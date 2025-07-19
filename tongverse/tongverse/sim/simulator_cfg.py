from __future__ import annotations

import dataclasses
from typing import Literal


class SimulatorCfg:
    @dataclasses.dataclass
    class PhysxCfg:
        """
        These parameters are used to configure the PhysX solver
        References:
            * PhysX 5 documentation: https://nvidia-omniverse.github.io/PhysX/
            * Orbit documentation:
            https://github.com/NVIDIA-Omniverse/orbit/blob/main/source
            /extensions/omni.isaac.orbit/omni/isaac/orbit/sim/simulation_cfg.py
        """

        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
        dt: float = 1.0 / 60.0
        """The physics time difference between each simulation step.
        Default is 0.0167 seconds."""

        substeps: int = 1
        """The number of physics simulation steps within one rendering step.
        Note:
            rendering means rendering a frame of the current application and not
        only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim
        will be refereshed as well if running non-headless."""

        use_gpu_pipeline: bool = False
        """Enable/disable GPU pipeline.
        If set to False, the physics data will be read as CPU buffers.
        """

        use_flatcache: bool = False
        use_gpu: bool = False
        """Enable/disable GPU accelerated dynamics simulation. Default is False.
        By default, broadphase type is MBP, if set to True,
        broadphase type would be GPU"""

        enable_scene_query_support: bool = False

        # GPU buffers
        gpu_max_rigid_contact_count: int = 2**23
        """Size of rigid contact stream buffer allocated in pinned host memory.
        Default is 2 ** 23."""

        gpu_max_rigid_patch_count: int = 5 * 2**15
        """Size of the rigid contact patch stream buffer allocated
        in pinned host memory. Default is 5 * 2 ** 15."""

        gpu_found_lost_pairs_capacity: int = 2**21
        """Capacity of found and lost buffers allocated in GPU global memory.
        Default is 2 ** 21.
        This is used for the found/lost pair reports in the BP"""

        gpu_found_lost_aggregate_pairs_capacity: int = 2**25
        """
        Capacity of found and lost buffers in aggregate system allocated
        in GPU global memory.Default is 2 ** 25.

        This is used for the found/lost pair reports in AABB manager
        """

        gpu_total_aggregate_pairs_capacity: int = 2**21
        """Capacity of total number of aggregate pairs allocated in GPU global memory.
        Default is 2 ** 21."""

        gpu_max_soft_body_contacts: int = 2**20
        """Size of soft body contacts stream buffer allocated in pinned host memory.
        Default is 2 ** 20."""

        gpu_max_particle_contacts: int = 2**20
        """Size of particle contacts stream buffer allocated in pinned host memory.
        Default is 2 ** 20."""

        gpu_heap_capacity: int = 2**26
        """Initial capacity of the GPU and pinned host memory heaps. Additional
        memory will be allocated if more memory is required. Default is 2 ** 26."""

        gpu_temp_buffer_capacity: int = 2**24
        """Capacity of temp buffer allocated in pinned host memory.
        Default is 2 ** 24."""

        gpu_max_num_partitions: int = 8
        """Limitation for the partitions in the GPU dynamics pipeline. Default is 8.

        This variable must be power of 2. A value greater than 32 is currently not
        supported. Range: (1, 32)
        """

        solver_type: Literal[0, 1] = 1
        """The type of solver to use.Default is 1 (TGS).

        Available solvers:

        * :obj:`0`: PGS (Projective Gauss-Seidel)
        * :obj:`1`: TGS (Truncated Gauss-Seidel)
        """
        enable_ccd: bool = True
        """Enable a second broad-phase pass that makes it possible to prevent objects
        from tunneling through each other. Default is False."""

        enable_stabilization: bool = True
        """Enable/disable additional stabilization pass in solver. Default is True."""

        bounce_threshold_velocity: float = 0.2
        """Relative velocity threshold for contacts to bounce (in m/s).
        Default is 0.2 m/s."""

        friction_offset_threshold: float = 0.04
        """Threshold for contact point to experience friction force (in m).
        Default is 0.04 m."""

        friction_correlation_distance: float = 0.025
        """Distance threshold for merging contacts into a single friction anchor point
        (in m). Default is 0.025 m."""

        use_fabric: bool = True
        """Enable/disable reading of physics buffers directly. Default is True.

        When running the simulation, updates in the states in the scene is normally
        synchronized with USD. This leads to an overhead in reading the data and does
        not scale well with massive parallelization. This flag allows disabling the
        synchronization and reading the data directly from the physics buffers.

        It is recommended to set this flag to :obj:`True` when running the simulation
        with a large number of primitives in the scene.

        Note:
            When enabled, the GUI will not update the physics parameters in real-time.
            To enable real-time updates, please set this flag to :obj:`False`.
        """

        disable_contact_processing: bool = False
        """Enable/disable contact processing. Default is False.

        By default, the physics engine processes all the contacts in the scene.
        However, reporting this contact information can be expensive due to its
        combinatorial complexity. This flag allows disabling the contact
        processing and querying the contacts manually by the user over a limited set of
        primitives in the scene.

        .. note::

            It is required to set this flag to :obj:`True` when using the TensorAPIs
            for contact reporting.
        """

    def __init__(self):
        self.sim_params = dataclasses.asdict(self.PhysxCfg())
        """Default is None"""
        self.backend: str = "torch"
        """the backend to be used (numpy or torch).
        Note:
        ORBIT only support the ``torch <https://pytorch.org/>`` backend for
        simulation. This means that all the data structures used in the simulation
        are ``torch.Tensor`` objects"""

        self.device: str = "cpu"
        """ Default is ``"None"``."""
        # FIXME: @Qi When setting the device to cuda:0, it is likely to encounter the
        # error: RunTimeError: Expected all tensors to be on the same device
        # but found at least two devices, cpu and cuda:0!


default_simulator_cfg = SimulatorCfg()
