from __future__ import annotations

from typing import Any, Callable, Optional


class ActuatorRegistryMeta(type):
    """
    Metaclass for task registration and instance creation.
    """

    def __init__(cls, name, bases, attrs):
        """
        Initialize the metaclass.

        :param name: Name of the class.
        :param bases: Base classes of the class.
        :param attrs: Attributes of the class.
        """
        super().__init__(name, bases, attrs)
        cls.mapping: dict = {}

    def register(
        cls, actuator_to_register: Optional[Any] = None, name: Optional[str] = None
    ) -> Callable:
        """
        Decorator for registering tasks or object states in the registry.

        :param task_to_register: The class to register.
        :param name: Optional name to register the item with.
        If None, the class/function name is used.
        :param task_type: Type of the task to register.
        :return: Callable function for registration.
        """

        def wrap(actuator_to_register):
            if not hasattr(actuator_to_register, "compute_effort"):
                raise AttributeError(
                    "{actuator_to_register} must have compute_effort() method"
                )

            register_name = actuator_to_register.__name__ if name is None else name
            cls.mapping[register_name] = actuator_to_register
            return actuator_to_register

        if actuator_to_register is None:
            return wrap
        return wrap(actuator_to_register)


class ActuatorRegistry(metaclass=ActuatorRegistryMeta):
    """
    Registry class for registering tasks and object states.
    """

    @classmethod
    def register_model(cls, actuator_to_register=None, *, name: Optional[str] = None):
        """
        Decorator for registering instruction-following tasks in the registry.

        :param to_register: The task class to register.
        :param name: Optional name to register the task with.
            If None, the class name is used.
        :return: Callable function for task registration.
        """
        return cls.register(actuator_to_register, name)

    @classmethod
    def create_instance(cls, agent, model_name: str):
        """
        Create a task instance based on the task configuration.

        :param task_config: Dictionary containing task configuration parameters.
        :return: Instance of the created task.
        """

        actuator_model = cls.mapping.get(model_name)
        if actuator_model is None:
            raise ValueError(f"Actuator type {model_name} is not registered.")
        return actuator_model(agent)


actuator_registry = ActuatorRegistry()
