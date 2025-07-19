from __future__ import annotations

from dataclasses import dataclass

from omni.isaac.core.materials.physics_material import PhysicsMaterial


@dataclass
class PhysicsMaterialCfg:
    """
    Data class representing the configuration for a physics material.

    Attributes:
        prim_path (str): The prim path of the physics material.
            Default is "/World/Physics_Materials/physics_material".
        static_friction (float): The static friction coefficient.
            Default is 0.5.
        dynamic_friction (float): The dynamic friction coefficient.
            Default is 0.5.
        restitution (float): The restitution coefficient.
            Default is 0.5.

    Note:
        The default values are the default values used by PhysX 5.
        For more details, refer to:
        https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#rigid-body-materials
    """

    prim_path: str = "/World/Physics_Materials/physics_material"
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.5


PHYSICSMATERIAL = PhysicsMaterial(
    prim_path=PhysicsMaterialCfg.prim_path,
    static_friction=PhysicsMaterialCfg.static_friction,
    dynamic_friction=PhysicsMaterialCfg.dynamic_friction,
    restitution=PhysicsMaterialCfg.restitution,
)
