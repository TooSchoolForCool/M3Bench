from __future__ import annotations

from pathlib import Path

ASSETS_VERSION = "v0.0.12"


class RootPath:
    """
    Class containing paths to various directories.

    Attributes:
        TONGVERSE (Path): Root path of the Tongverse project.
        AGENT (Path): Path to the agent assets directory.
        SCENE (Path): Path to the scene assets directory.
        CHARACTER (Path): Path to the character assets directory.
        VKC_REQUEST_LOG (Path): Path to the VKC request log directory.
        VKC_RESPONSE_LOG (Path): Path to the VKC response log directory.
        LIVEDEMO (Path): Path to the live demo directory.
    """

    TONGVERSE = Path(__file__).parent.parent
    AGENT = TONGVERSE / "data" / "assets" / ASSETS_VERSION / "agent"
    SCENE = TONGVERSE / "data" / "assets" / ASSETS_VERSION / "scene"
    CHARACTER = TONGVERSE / "data" / "assets" / ASSETS_VERSION / "character"
    VKC_REQUEST_LOG = TONGVERSE.parent / "log" / "vkc" / "request_log"
    VKC_RESPONSE_LOG = TONGVERSE.parent / "log" / "vkc" / "response_log"
    LIVEDEMO = TONGVERSE.parent / "docs" / "livedemo"
    IMG = TONGVERSE.parent / "sensor_output"


class TmpPath:
    """
    This class holds temporary prefix prim path created during the task
    """
    FIXED_JOINT_PREFIX: str = "/World/tmp/fixed_joint"
