from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple, Union

from omni.isaac.core.prims import RigidPrim, XFormPrim
from omni.isaac.core.utils.prims import delete_prim
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.stage import add_reference_to_stage
from torch import Tensor

# pylint: enable=wrong-import-position
from tongverse.object import ArticulatedObject, BaseObject, RigidObject
from tongverse.scene.scene_registry import SceneRegistry
from tongverse.utils import deletable_prim_check
from tongverse.utils.constant import RootPath


class BaseScene:
    """Provides high-level functionalities for managing scenes.

    This class provides functions to:
        - Load scene and set scene position
        - Manage the objects in the scene through SceneRegistry
        - Preprocess objects information from obj_simu_cfg.json and attachments.json
        - Create objects and add them to the scene registry based on their configuration
        - Add semantic labels to objects

    Args:
        scene_cfg (dict): Scene configuration.
            The dictionary should contain the following keys:
                - name (str, optional): Name of the scene. Default is "House_22".
                - position ( Sequence[float], optional): Position of the scene.
                Default is (0, 0, 0).
                - orientation (Sequence[float], optional): Orientation of the scene in
                quaternion format (w, x, y, z).
                    Default is (1, 0, 0, 0).
                - obj_semantic_label (str, optional): Semantic label to be added to
                objects. Default is "baselink".
    """

    def __init__(self, scene_cfg: dict) -> None:
        """Initialize BaseScene with the provided configuration.

        Args:
            scene_cfg (dict): A dictionary containing the scene configuration.

        Raises:
            TypeError: If `scene_cfg` is not a dictionary.
        """
        if not isinstance(scene_cfg, dict):
            raise TypeError("scene_cfg must be a dictionary")
        self._scene_cfg = scene_cfg
        self._scene_name = None
        self._prim = None
        self._obj_semantic_label = None
        self._scene_registry = SceneRegistry()
        self._load = False
        self.load()

    def load(self) -> None:
        """Load the scene and objects into the stage.

        This method processes the scene configuration and object information,
        then adds them to the scene registry. After loading, the `_load` flag
        is set to True.
        """
        self._process_scene_cfgs()
        self._process_obj_cfgs()
        self._load = True

    def initialize_physics(self) -> None:
        """Initialize physics for objects in the scene.
        Before initializing scene physics, ensure that the simulator physics has been
        initialized
        (by calling env.reset()). Failure to follow this order may result in a
        RuntimeError.

        After initializing scene physics, this method iterates through all objects in
        the scene registry, initializes their physics, and sets them to
        their default state.
        """
        if not self._load:
            self.load()

        # 1. Initialize scene physics
        self._prim.initialize()
        self._prim.post_reset()

        for obj_registery in self._scene_registry.all_registries_objects:
            for obj_name in list(obj_registery):
                if not obj_registery[obj_name].baselink.is_valid():
                    self._scene_registry.remove_object(name=obj_name)
                else:
                    # 2. Initialize the object's physics,
                    # and set its default state.
                    obj_registery[obj_name].initialize_physics()

        self._load = True

    def remove(self):
        if deletable_prim_check(self.prim_path):
            delete_prim(self.prim_path)

    def get_object_by_name(
        self, name: str
    ) -> Optional[Union[BaseObject, RigidObject, ArticulatedObject]]:
        """Retrieve an object from the scene registry by its name.

        Args:
            name (str): The name of the object to retrieve.

        Returns:
            Union[BaseObject, RigidObject, ArticulatedObject] or None: The object with
            the specified name,
            or None if not found.
        """

        return self._scene_registry.get_object(name=name)

    def get_all_articulated_objects(self) -> dict:
        """Retrieve all articulated objects from the scene registry.

        Returns:
            dict: A dictionary containing all articulated objects in the scene registry.
        """
        return self._scene_registry.articulated_objects

    def get_all_static_objects(self) -> dict:
        """Retrieve all static objects from the scene registry.

        Returns:
            dict: A dictionary containing all static objects in the scene registry.
        """
        return self._scene_registry.static_objects

    def get_all_rigid_objects(self) -> dict:
        """Retrieve all rigid objects from the scene registry.

        Returns:
            dict: A dictionary containing all rigid objects in the scene registry.
        """
        return self._scene_registry.rigid_objects

    def get_objects_by_category(
        self, category: str
    ) -> List[Union[BaseObject, RigidObject, ArticulatedObject]]:
        """Retrieve objects from the scene registry based on their category.

        Args:
            category (str): The category of objects to retrieve.

        Returns:
            List[Union[BaseObject, RigidObject, ArticulatedObject]]: A list of objects
            belonging to the specified category.
        """
        return self._scene_registry.get_objects_by_category(category)

    def get_all_objects(self) -> dict:
        """Retrieve all objects from the scene registry.

        Returns:
            dict: A dictionary containing all objects in the scene registry,
            where the keys are the object names and the values are the objects
            themselves.
        """
        res = {}
        for object_dict in self._scene_registry.all_registries_objects:
            for obj in object_dict.values():
                res[obj.name] = obj
        return res

    def get_world_pose(self) -> Tuple[Tensor, Tensor]:
        """Retrieve the world pose of the scene.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the positions and orientations of
            the scene objects.
        """
        positions, orientations = self._prim.get_world_pose()
        return positions, orientations

    def set_world_pose(
        self,
        pos: Optional[Sequence[float]] = None,
        orient: Optional[Sequence[float]] = None,
    ):
        """Set the world pose of the scene.

        Args:
            pos (Optional[Sequence[float]]): Position. Default is None.
            orient (Optional[Sequence[float]]: Quaternion Orientation is scalar-first
            (w, x, y, z). shape is (4, ). Default is None.
        """
        self._prim.set_world_pose(position=pos, orientation=orient)

    @property
    def name(self) -> str:
        """Retrieve the name of the scene.

        Returns:
            str: The name of the scene.
        """
        return self._scene_name

    @property
    def prim(self) -> RigidPrim:
        """Retrieve the RigidPrim of the scene.

        Returns:
            RigidPrim: The RigidPrim object representing the scene.
        """
        return self._prim

    @property
    def prim_path(self) -> str:
        return self._prim.prim_path

    @property
    def object_semantic_label(self) -> str:
        """Retrieve the semantic label of objects in the scene.

        Returns:
            str: The semantic label used for objects in the scene.
        """
        return self._obj_semantic_label

    # == Internal Helper ==
    def _process_scene_cfgs(self) -> None:
        """Process the scene configuration.

        This method extracts necessary information from the scene configuration
        provided during initialization,
        such as the scene name, spawn position, orientation, and object semantic label.
        It then loads the scene into the stage with the specified USD file path,
        creates a RigidPrim representing the scene, and sets its initial world pose.

        NOTE: Simulation of multiple RigidBodyAPI's in a hierarchy will cause
        unpredicted results.Thus, prim of scene is initialized as XFormPrim
        instead of RigidPrim
        """
        self._scene_name = self._scene_cfg.get("name", "House_22")
        spawn_position = self._scene_cfg.get("position", (0, 0, 0))
        spawn_orientation = self._scene_cfg.get("orientation", (1, 0, 0, 0))
        self._obj_semantic_label = self._scene_cfg.get("obj_semantic_label", "baselink")

        prim_path = f"/World/Scene/{self._scene_name}"
        usd_file_path = str(RootPath.SCENE / self._scene_name / "main.usd")

        add_reference_to_stage(usd_path=usd_file_path, prim_path=prim_path)
        self._prim = XFormPrim(
            prim_path=prim_path,
            name=self._scene_name,
        )
        self.set_world_pose(spawn_position, spawn_orientation)

    def _read_object_metadata(self) -> list[dict]:
        """Read object metadata from JSON files and return a dictionary containing
        object configurations.

        Returns:
            list[dict]: A list of dictionaries representing object configurations.
            e.g.
                [{
                    "link": ["baseballbat_3_link"],
                    "baselink": "baseballbat_3_link",
                    "rigidbody": true,
                    "enable_collision": true,
                    "category": "baseballbat",
                    "articulated": false
                    "attachments":[]
                    }]
        """
        metadata_file_path = str(
            RootPath.SCENE / self._scene_name / "obj_simu_cfg.json"
        )
        vkc_attachments_path = str(
            RootPath.SCENE / self._scene_name / "attachments.json"
        )

        obj_dict = {}
        with open(metadata_file_path, encoding="utf-8") as f:
            data = json.load(f)
            obj_dict = data["objects"]

        try:
            with open(vkc_attachments_path, encoding="utf-8") as f:
                data = json.load(f)
                for base_link, info in data.items():
                    assert base_link in obj_dict, (
                        f"The base link {base_link} from attachments.json "
                        "does not match with obj_simu_cfg.json."
                    )

                    obj_dict[base_link]["attachments"] = info["attachments"]
        except FileNotFoundError:
            print(
                f"The file {vkc_attachments_path} does not exist. "
                "You can add VKC attachment by calling object.add_attachment() method "
                "if using VKC planner."
            )
        return obj_dict.values()

    def _add_object_to_scene_registry(self, obj_cfg_list: list) -> None:
        """Add objects to the scene registry based on their configurations.

        Currently supports 'static', 'articulated', and 'rigid' objects.
        Support for 'particle', 'cloth', and 'deformable' objects may be added in
        future releases.

        Args:
            obj_cfg_list (list): A list of dictionaries representing object
            configurations.

        Raises:
            TypeError: If the object type is not supported.
        """
        for obj_cfg in obj_cfg_list:
            if obj_cfg.get("articulated", False):
                obj = ArticulatedObject(obj_cfg, self.prim.prim_path)
                self._scene_registry.add_articulated_object(obj)
            elif obj_cfg.get("rigidbody", False):
                obj = RigidObject(obj_cfg, self.prim.prim_path)
                self._scene_registry.add_rigid_object(obj)
            elif not obj_cfg.get("rigidbody", False):
                obj = BaseObject(obj_cfg, self.prim.prim_path)
                self._scene_registry.add_static_object(obj)
            else:
                raise TypeError(
                    f"The object type of {obj_cfg.get('baselink')} is not supported at "
                    "this time."
                )

    def _add_object_semantic_label(self) -> None:
        """Add semantic labels to objects in the scene registry.

        This method iterates through all objects in the scene registry and adds a
        semantic label to each prim link of the object based on the specified
        semantic label.
        """
        if self._obj_semantic_label not in ["baselink", "category"]:
            raise ValueError(
                f"{self._obj_semantic_label} is neither baselink nor " f"category"
            )
        for prim_registery in self._scene_registry.all_registries_objects:
            for prim_name in list(prim_registery):
                prim_object = prim_registery[prim_name]
                if self._obj_semantic_label == "baselink":
                    label = prim_object.name
                else:
                    label = prim_object.category
                add_update_semantics(
                    prim=prim_object.baselink.prim,
                    semantic_label=label,
                    type_label="class",
                )
                if isinstance(prim_object, ArticulatedObject):
                    for prim in prim_object.links_dict.values():
                        add_update_semantics(
                            prim=prim.prim,
                            semantic_label=label,
                            type_label="class",
                        )

    def _process_obj_cfgs(self) -> None:
        """Process object configurations.

        This method preprocesses object information from JSON files, adds the objects
        into the scene registry based on their configuration, and adds semantic labels
        to each object in the scene registry.

        Note:
            This method relies on the `_read_object_metadata()` and `_
            add_object_to_scene_registry()` methods.
        """
        obj_dict = self._read_object_metadata()
        self._add_object_to_scene_registry(obj_dict)
        self._add_object_semantic_label()
