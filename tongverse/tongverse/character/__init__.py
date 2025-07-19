from __future__ import annotations

from tongverse.character.character import Character
from tongverse.character.character_cfg import CharacterCfg


def get_baby_config() -> CharacterCfg:
    cfg = CharacterCfg()
    cfg.name = "AIBaby"
    cfg.position = (0, 0, 0)
    cfg.orientation = (1, 0, 0, 0)
    cfg.scale = 0.01
    cfg.head_name = "head"
    cfg.eye_position_wrt_head = (0.0, 8.6, -7.0)
    cfg.eye_orientation_wrt_head = (0.5, -0.5, -0.5, -0.5)
    return cfg


def get_male_config() -> CharacterCfg:
    cfg = CharacterCfg()
    cfg.name = "AIMale"
    cfg.position = (0, 0, 0)
    cfg.orientation = (1, 0, 0, 0)
    cfg.scale = 0.01
    cfg.head_name = "head"
    cfg.eye_position_wrt_head = (8.5, 0.0, 0.0)
    cfg.eye_orientation_wrt_head = (0.5, 0.5, 0.5, 0.5)
    return cfg
