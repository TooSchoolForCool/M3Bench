from __future__ import annotations

from typing import List, Optional, Union

from tongverse.object import ArticulatedObject, BaseObject, RigidObject


class SceneRegistry:
    """
    A registry for managing objects in a scene.
    """

    def __init__(self) -> None:
        """
        Initialize the SceneRegistry.
        """
        self._rigid_objects = {}
        self._static_objects = {}
        self._articulated_objects = {}
        self._all_registries_objects = [
            self._rigid_objects,
            self._static_objects,
            self._articulated_objects,
        ]

    @property
    def rigid_objects(self) -> dict:
        """
        Get the dictionary of rigid objects.
        """
        return self._rigid_objects

    @property
    def static_objects(self) -> dict:
        """
        Get the dictionary of static objects.
        """
        return self._static_objects

    @property
    def articulated_objects(self) -> dict:
        """
        Get the dictionary of articulated objects.
        """
        return self._articulated_objects

    @property
    def all_registries_objects(self) -> List[dict]:
        """
        Get a list containing all object dictionaries.
        """
        return self._all_registries_objects

    def add_rigid_object(self, obj: RigidObject) -> None:
        """
        Add a rigid object to the scene.

        Parameters:
        obj (RigidObject): The rigid object to add to the scene.

        Raises:
        ValueError:  If the object's name is already in use.
        """
        if self.name_exists(obj.name, self._rigid_objects):
            raise ValueError(
                f"The object {obj.name} cannot be added to the scene because "
                "its name is already in use."
            )
        self._rigid_objects[obj.name] = obj

    def add_static_object(self, obj: BaseObject) -> None:
        """
        Add a static object to the scene.

        Parameters:
        obj (BaseObject): The static object to add to the scene.

        Raises:
        ValueError:  If the object's name is already in use.
        """
        if self.name_exists(obj.name, self._static_objects):
            raise ValueError(
                f"The object {obj.name} cannot be added to the scene because "
                "its name is already in use."
            )
        self._static_objects[obj.name] = obj

    def add_articulated_object(self, obj: ArticulatedObject) -> None:
        """
        Add an articulated object to the scene.

        Parameters:
        obj (ArticulatedObject): The articulated object to add to the scene.

        Raises:
        ValueError:  If the object's name is already in use.
        """
        if self.name_exists(obj.name, self._articulated_objects):
            raise ValueError(
                f"The object {obj.name} cannot be added to the scene because "
                "its name is already in use."
            )
        self._articulated_objects[obj.name] = obj

    def get_object(
        self, name: str
    ) -> Optional[Union[BaseObject, RigidObject, ArticulatedObject]]:
        """
        Get an object from the scene by its name.

        Parameters:
        name (str): The name of the object to retrieve.

        Returns:
        Optional[Union[BaseObject, RigidObject, ArticulatedObject]]:
        The object with the specified name, or None if not found.
        """
        for object_dict in self._all_registries_objects:
            if name in object_dict:
                return object_dict[name]
        return None

    def get_objects_by_category(
        self, category: str
    ) -> List[Union[BaseObject, RigidObject, ArticulatedObject]]:
        """
        Get all objects from the scene with a specified category.

        Parameters:
        category (str): The category of objects to retrieve.

        Returns:
        List[Union[BaseObject, RigidObject, ArticulatedObject]]:
        A list of objects with the specified category.
        """
        # TODO: optimize the nested loop
        res = []
        for object_dict in self._all_registries_objects:
            for obj in object_dict.values():
                if obj.category == category:
                    res.append(obj)
        return res

    def remove_object(self, name: str) -> None:
        """
        Remove an object from the scene by its name.

        Parameters:
        name (str): The name of the object to remove.

        Raises:
        RuntimeError: If the object with the specified name does not exist in the scene.
        """
        for object_dict in self._all_registries_objects:
            if name in object_dict:
                del object_dict[name]
                return

        raise RuntimeError(
            f"The object {name} cannot be removed from the scene as it does not exist."
        )

    def name_exists(self, name: str, object_dict: dict) -> bool:
        """
        Check if a name exists in a given dictionary.

        Parameters:
        name (str): The name to check.
        object_dict (dict): The dictionary to search for the name.

        Returns:
        bool: True if the name exists in the dictionary, False otherwise.
        """
        return name in object_dict
