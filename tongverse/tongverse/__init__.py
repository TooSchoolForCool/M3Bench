from __future__ import annotations

__version__ = "0.0.2"
import logging
import sys

import carb
from omni.isaac.kit import SimulationApp


def enable_required_extensions() -> None:
    """
    Enable additional extensions required for the TongVerse.

    Extensions:
    - omni.anim.people
    - omni.anim.graph.core
    - omni.kit.window.viewport (for Windows)
    - omni.kit.scripting
    - omni.kit.asset_converter (for FBX asset conversion)
    - omni.kit.window.movie_capture (for screen capture)
    """
    extensions = [
        "omni.anim.people",
        "omni.anim.graph.core",
        "omni.kit.window.viewport",
        "omni.kit.scripting",
        "omni.kit.asset_converter",
        "omni.kit.window.movie_capture",
    ]
    # pylint: disable=import-outside-toplevel
    from omni.isaac.core.utils.extensions import enable_extension

    for extension in extensions:
        enable_extension(extension)


def configure_logging() -> None:
    """
    Configure logging to suppress unnecessary warnings.
    FIXME: @QI: There seems to be an issue with deactivating the warning;
    it still doesn't work properly.
    """

    loggers = [
        "omni.hydra",
        "omni.isaac.urdf",
        "omni.physx.plugin",
    ]
    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    carb.settings.get_settings().set_string("/log/level", "Error")
    carb.settings.get_settings().set_string("/log/fileLogLevel", "Error")
    carb.settings.get_settings().set_string("/log/outputStreamLevel", "Error")


def start() -> bool:
    """
    Start the TongVerse.

    Returns:
        bool: True if the TongVerse started successfully, False otherwise.
    """
    try:
        enable_required_extensions()
        configure_logging()
        return True
    except RuntimeError as e:
        logging.error("Failed to start simulation: %s", e)
        return False


def shutdown():
    """
    Shutdown and exit the application.
    """
    app.close()
    # clean exit
    sys.exit(0)


app = SimulationApp(
    {"headless": False, "renderer": "RayTracedLighting", "multi_gpu": True}
)

start()
