from __future__ import annotations

import gc
from typing import Optional

import omni.kit.commands
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.utils.prims import (
    delete_prim,
    get_prim_at_path,
    get_prim_children,
    get_prim_path,
)
from omni.isaac.core.utils.types import ArticulationAction

from tongverse.agent import get_agent
from tongverse.agent.base_agent import Agent
from tongverse.data import download_assets_if_not_existed

# pylint: enable=wrong-import-position
from tongverse.scene.base_scene import BaseScene
from tongverse.sim.simulator import Simulator
from tongverse.utils import deletable_prim_check
from tongverse.utils.constant import TmpPath
from tongverse.utils.material import PHYSICSMATERIAL

from .env_cfg import EnvCfg, default_env_cfg

ENV_STATE_API_VER = "0.3"


class Env:
    """
    This class provides functions to:
        - Managing the simulation environment.
        - Loading scenes, agents, and tasks.
        - Configuring lighting and ground settings.

    Note:
        If using the VKC planner, only one scene and one agent can be loaded
        into the environment.
    """

    def __init__(self, config: EnvCfg = default_env_cfg) -> None:
        """
        Initializes the environment.

        Parameters:
            config (EnvCfg, optional): The configuration for the environment.
            Defaults to the configuration from EnvCfg.

        Note:
            env.reset() should be called after scene, agent, tasks are loaded
        """
        self.config = config
        self.sim = Simulator(config.simulator_cfg)
        self._agent = None
        self._scene = None
        self._load()
        self._vkc_planner = True

    def _set_view(self) -> None:
        """
        Configures the environment view settings,
        including lighting and ground properties(with physics material).
        """

        omni.kit.commands.execute(
            "SetLightingMenuModeCommand",
            lighting_mode=self.config.view_cfg["light_mode"],
        )

        ground_cfg = self.config.ground_cfg
        GroundPlane(
            prim_path=ground_cfg["prim_path"],
            name=ground_cfg["name"],
            z_position=ground_cfg["z_position"],
            visible=ground_cfg["visible"],
            physics_material=PHYSICSMATERIAL,
        )

    def _load_scene(self, scene_cfg: dict) -> None:
        """
        Loads a scene based on the provided scene configuration.

        Parameters:
            scene_cfg (dict): Configuration for the scene.

        Raises:
            TypeError: If scene_cfg is not a dictionary or is None.
        """
        if not isinstance(scene_cfg, dict):
            raise TypeError("scene_cfg must be a dictionary")
        self._scene = BaseScene(scene_cfg)

    def _load_agent(self, agent_cfg: dict) -> None:
        """
        Loads an agent based on the provided agent configuration.

        Parameters:
            agent_cfg (dict): Configuration for the agent.

        Raises:
            TypeError: If agent_cfg is not a dictionary or is None.
        """

        if not isinstance(agent_cfg, dict):
            raise TypeError("agent_cfg must be a dictionary")

        self._agent = get_agent(agent_cfg)

    def _load(self) -> None:
        """
        Loads scenes, agents, and configures the view.
        """
        download_assets_if_not_existed()
        if self.config.scene is not None:
            self._load_scene(self.config.scene)
        if self.config.agent is not None:
            self._load_agent(self.config.agent)

        self._set_view()

    def reload_scene(self, scene_cfg: dict):
        """
        Remove current scene usd and reload a new one.

        NOTE:
            Remember to call env.reset() after loading the new scene.

        Parameters:
            scene_cfg(dict): Configuration for the new scenne
        """
        self.remove_scene()
        self._load_scene(scene_cfg)

    def remove_scene(self):
        self.sim.stop()
        # remove scene
        if self._scene is not None:
            self._scene.remove()
            self._scene = None
            gc.collect()

    def reload_agent(self, agent_cfg: dict):
        """
        Remove current agent usd and reload a new one.

        NOTE:
            Remember to call env.reset() after loading the new agent

        Parameters:
            agent_cfg(dict): Configuration for the new agent
        """
        self.remove_agent()
        self._load_agent(agent_cfg)

    def remove_agent(self):
        self.sim.stop()
        # remove agent
        if self._agent is not None:
            self._agent.remove()
            self._agent = None
            gc.collect()

    def step(
        self,
        target_action: ArticulationAction = None,
        render: bool = True,
    ) -> None:
        """
        Advances the simulation by one step.

        Parameters:
            render (bool): Whether to render the simulation.
            Default is True.

        Note:
            If set render to False, the application will
            not be rendered but keep simulating physics
        """
        if self._agent:
            self._agent.apply_action(target_action)
        self.sim.step(render)
        # must update agent buffer after step, according to Orbit
        if self._agent:
            self._agent.update(self.sim.get_physics_dt())

    def reset(self, soft: bool = False) -> None:
        """
        Resets the environment by resetting the physics simulation,
        scene objects and agents.

        Parameters:
            soft (bool, optional): If True, timeline won't be stopped and played again
            (stop + Play on the timeline).
            Default is False,


        Note:
          Internal buffers of agents (orbit) will be cleared when calling sim.reset().


        """
        # Check if there are newly added fixed joints (attachment)
        # during the task running.
        if soft is False:
            # delete all newly added joints
            tmp_fixed_joint_prefix_path = f"{TmpPath.FIXED_JOINT_PREFIX}"
            if deletable_prim_check(tmp_fixed_joint_prefix_path):
                prim = get_prim_at_path(tmp_fixed_joint_prefix_path)
                for child_prim in get_prim_children(prim):
                    delete_prim(get_prim_path(child_prim))

        # need to initialize physics getting any articulation..etc
        self.sim.reset(soft=soft)

        # when call sim.reset(), agent should be reset as well
        # to set back to default state and clear internal buffers
        if self._scene is not None:
            self._scene.initialize_physics()
        if self._agent is not None:
            self._agent.reset()

    def pause(self) -> None:
        """Pause the physics simulation"""
        self.sim.pause()

    def get_agent(self) -> Optional[Agent]:
        """
        Retrieves the agent based on its name.

        Parameters:
            agent_name (str, optional): The name of the agent to retrieve.
                If None, returns the first agent loaded.

        Returns:
            Agent (Agent, Optional): The agent corresponding to the provided name,
                or the first agent loaded if name is None.
                Returns None if there are no agents in the environment.
        """
        return self._agent

    def get_scene(self) -> Optional[BaseScene]:
        """
        Retrieves the scene based on its name.

        Parameters:
            scene_name (str, optional): The name of the scene to retrieve.
                If None, returns the first scene loaded.

        Returns:
            Scene (BaseScene, Optional): The scene corresponding to the provided name,
                or the first scene loaded if name is None.
                Returns None if there are no scenes in the environment.
        """
        return self._scene

    def get_env_state(self) -> dict:
        """
        Retrieves the current state of objects and agents in the current simulation
        environment.

        Returns:
            dict:
                - api_version: The current ENV_STATE_API_VER.
                - timestamp: The time when this method is called.
                - base_link_state (dict): Contains the position and orientation of all
                   objects' baselink. The key is the object's name, and the value is a
                   list [X, Y, Z, QW, QX, QY, QZ].
                - joint_state (dict): Contains the joint position of all articulated
                    objects and agents. The key is the joint name, and the value is
                    the joint current position.

        Note:
            - Fixed joint information won't be returned
            - The orientation representation of object is in quaternion format
            (QW, QX, QY, QZ).

        Raises:
            RuntimeError: If VKC planner is being used but more than one scene or agent
            is loaded.

        TODO:
            Add agent information
        """
        # if self._vkc_planner and len(self._scenes) > 1:
        #     raise RuntimeError(
        #         "VKC only supports loading one scene at a time. Currently, there are "
        #         "more than one scene loaded."
        #     )
        # if self._vkc_planner and len(self._agents) > 1:
        #     raise RuntimeError(
        #         "VKC only supports loading one agent at a time. Currently, there are "
        #         "more than one agent loaded."
        #     )

        # res = {}
        # res["api_version"] = ENV_STATE_API_VER
        # res["timestamp"] = time.time()
        # res["joint_state"] = {}
        # scene = self.get_scene()
        # base_link_state = {}

        # for obj_name, obj in scene.get_all_objects().items():
        #     pos = obj.get_world_pose()[0].cpu().numpy().tolist()
        #     orient = obj.get_world_pose()[1].cpu().numpy().tolist()
        #     base_link_state[obj_name] = [
        #         pos[0],
        #         pos[1],
        #         pos[2],
        #         orient[1],
        #         orient[2],
        #         orient[3],
        #         orient[0],
        #     ]
        #     if hasattr(obj, "get_joint_positions"):
        #         res["joint_state"].update(obj.get_joint_positions())

        # res["base_link_state"] = base_link_state

        # get agent current dof state
        # agents = self.get_agent()
        # for agent in agents:
        #     if agent is not None:
        #         res["joint_state"].update(agent.get_joint_positions())

        # return res
