from __future__ import annotations

from typing import Optional, Sequence


class CharacterCfg:
    """
    Configuration for a character (skeleton) within the environment.
    """

    name: str = "AIBaby"

    position: Optional[Sequence[float]] = (0, 0, 0)
    orientation: Optional[Sequence[float]] = (1, 0, 0, 0)
    # Skeleton converted from UE holds 100 meters per unit by default.
    scale: Optional[float] = 0.01

    head_name: str = "head"
    eye_position_wrt_head: Optional[Sequence[float]] = (0.0, 8.6, -7.0)
    eye_orientation_wrt_head: Optional[Sequence[float]] = (0.5, -0.5, -0.5, -0.5)
