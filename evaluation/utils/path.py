from pathlib import Path

ASSETS_VERSION = "v0.0.6"

class RootPath:
    VKC_DEPS = Path("/env/")
    AGENT = VKC_DEPS / "tongverse_agents" / "agent"
    SCENE = VKC_DEPS / "physcene"