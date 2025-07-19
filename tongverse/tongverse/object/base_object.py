from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import get_prim_children
from torch import Tensor


class BaseObject:
    def __init__(self, obj_cfg: dict, path_prefix: str = "/World/Scene") -> None:
        """
        Initializes a BaseObject instance during scene loading.

        Parameters:
            path_prefix (str, optional): The prefix for the path to the object
            in the scene hierarchy. Defaults to "/World/Scene".
            obj_cfg (dict): The configuration for the object. This information is
            obtained after processing the metadata file.

        Raises:
            TypeError: If obj_cfg is None or not a dictionary.

        NOTE:
            An object can consist of multiple links,
            thus a dictionary called "links_dict" is used to store its link information,
            where the key represents the link name and the value corresponds to the
            link prim.
        """

        if obj_cfg is None or not isinstance(obj_cfg, dict):
            raise TypeError("obj_cfg must be a dictionary")
        self._path_prefix = path_prefix
        self._obj_cfg = obj_cfg
        self._obj_name = None
        self._category = None
        self._baselink = None
        self._attachments = {}
        self._process_cfgs()

    def initialize_physics(self) -> None:
        """
        This method iterates through all prim links of this object,
        initializes physics, and resets each prim link to its default state
        (positions and orientations).
        """
        self._baselink.initialize()
        self._baselink.post_reset()

    def update_visual_materials(
        self, visual_materials: VisualMaterial, weaker_than_descendants: bool
    ):
        """Apply visual material to the links and optionally their link descendants.

        Args:
            visual_materials (Union[VisualMaterial, List[VisualMaterial]]):
            visual materials to be applied to the links. Currently supports
            PreviewSurface, OmniPBR and OmniGlass. If a list is provided then
            its size has to be equal the view's size or indices size.
            If one material is provided it will be applied to all links in the view.
            weaker_than_descendants (Optional[Union[bool, List[bool]]], optional):
            True if the material shouldn't override the descendants
            materials, otherwise False. Defaults to False.
            If a list of visual materials is provided then a list
            has to be provided with the same size for this arg as well.
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]],
              optional):
            indices to specify which links
            to manipulate. Shape (M,).
            Where M <= size of the encapsulated links in the view.
            Defaults to None (i.e: all links in the view).

        Raises:
            Exception: length of visual materials != length of links indexed
            Exception: length of visual materials != length of weaker descendants
            boolsarg

        Example:

        .. code-block:: python

            >>> from omni.isaac.core.materials import OmniGlass
            >>>
            >>> # create a dark-red glass visual material
            >>> material = OmniGlass(
            ...     link_path="/World/material/glass",
            # path to the material link to create
            ...     ior=1.25,
            ...     depth=0.001,
            ...     thin_walled=False,
            ...     color=np.array([0.5, 0.0, 0.0])
            ... )
            >>> links.apply_visual_materials(material)
        """
        visual_mesh = get_prim_children(self._baselink)[0]

        assert (
            visual_mesh.GetName() == "visuals"
        ), "There's no visual mesh for this object"

        visual_mesh.apply_visual_materials(visual_materials, weaker_than_descendants)

    def get_applied_visual_materials(self) -> VisualMaterial:
        """
        Retrieves the currently applied visual materials to the links
        if their type is supported.

        Returns:
            VisualMaterial: A list of the current applied visual materials.

        """
        return self._baselink.get_applied_visual_materials()

    def get_diffuse_color(self):
        pass

    def set_diffuse_color(self):
        pass

    def get_attachment(self, attach_name: str = None):
        pass

    def add_attachment(self, att_dict: dict) -> bool:
        pass

    def update_attachment(
        self, attach_name: str, update_key: str, update_value: Union[str, List[float]]
    ) -> bool:
        pass

    def get_visibility(self) -> bool:
        """
        Retrieves the visibility status of the object

        Returns:
            bool: True if the link is visible in the stage, False otherwise.
        """
        return self._baselink.get_visibility()

    def set_visibility(self, val: bool) -> None:
        """
        Sets the visibility of the object.

        Args:
            val (bool): A flag to set the visibility.
        """
        self._baselink.set_visibility(val)

    def set_world_pose(
        self,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Sets the world pose (position and orientation) of the object.

        Args:
            position (Optional[Sequence[float]], optional): The position to set the
            object to. Defaults to None.
            orientation (Optional[Sequence[float]], optional): The orientation to
            set the object to. Defaults to None.
        """
        if position is not None:
            self._baselink.set_world_pose(position=position)

        if orientation is not None:
            self._baselink.set_world_pose(orientation=orientation)

    def get_world_pose(self) -> Tuple[Tensor, Tensor]:
        """
        Retrieves the world pose (position and orientation) of the object.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the positions and orientations
            of the object.
        """
        positions, orientations = self._baselink.get_world_pose()
        return positions, orientations

    def set_local_pose(
        self,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ):
        """Set link's pose with respect to the local frame (the link's parent frame).

        .. warning::

            This method will change (teleport) the link pose immediately to the
            indicated value

        Args:
            translation (Optional[Sequence[float]], optional): translation in the local
            frame of the link
            (with respect to its parent link). shape is (3, ).
            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation
            in the local frame of the link.
            quaternion is scalar-first (w, x, y, z). shape is (4, ).
            Defaults to None, which means left unchanged.
        .. hint::

            This method belongs to the methods used to set the link state

        Example:

        .. code-block:: python

            >>> link.set_local_pose(translation=np.array([1.0, 0.5, 0.0]),
            orientation=np.array([1., 0., 0., 0.]))
        """
        if translation is not None:
            self._baselink.set_local_pose(translation=translation)

        if orientation is not None:
            self._baselink.set_local_pose(orientation=orientation)

    def get_local_pose(self) -> Tuple[Tensor, Tensor]:
        positions, orientations = self._baselink.get_local_pose()
        return positions, orientations

    @property
    def name(self) -> str:
        """
        Retrieves the name of the object.

        Returns:
            str: The name of the object.
        """
        return self._obj_name

    @property
    def category(self) -> str:
        """
        Retrieves the category of the object.

        Returns:
            str: The category of the object.
        """
        return self._category

    @property
    def baselink_path(self) -> str:
        """
        Retrieves the path of the root link of the object.

        Returns:
            str: The path of the root link of the object.
        """
        return self._baselink.prim_path

    @property
    def baselink(self) -> XFormPrim:
        """
        Retrieves the base link of the object.

        Returns:
            XFormPrim: The root link of the object.
        """
        return self._baselink

    # ==========================internal helper=================================
    def _process_cfgs(self) -> None:
        """
        Process the object configuration.

        This method creates XFormPrim instances for each link specified
        in the configuration.
        If the object has attachments, it adds them using the add_attachment method.
        """
        self._obj_name = self._obj_cfg.get("baselink", "xform_obj")
        self._baselink = XFormPrim(
            f"{self._path_prefix}/{self._obj_name}", self._obj_name
        )
        self._category = self._obj_cfg.get("category")

        # if this object has vkc attachment
        if "attachments" in self._obj_cfg:
            for att in self._obj_cfg["attachments"]:
                self.add_attachment(att)
                # self._attachments[att["attach_name"]] = VKCAttachment(**att)
