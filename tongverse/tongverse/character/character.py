from __future__ import annotations

import os
from typing import Iterable, List, Optional, OrderedDict, Sequence

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.prims import XFormPrim
from pxr import Gf, Sdf, UsdGeom, UsdSkel

from tongverse.character.character_cfg import CharacterCfg
from tongverse.character.interface.animation_interface import AnimationInterface
from tongverse.character.interface.skeleon_interface import SkeletonInterface
from tongverse.sensor import Camera, default_camera_cfg
from tongverse.utils.constant import RootPath


class Character:
    """
    Character controller class
    """

    PRIM_PREFIX = "/World/Characters/"

    def __init__(self, character_cfg: CharacterCfg) -> None:
        """Initialize the AICharacter class.

        It is not recommended to call this function directly since it requires the user
        to load the character manually. Instead, use the `get_character` outside of this
        class to get the character object.

        Args:
            env (tongverse.env.Env): environment object
            character_cfg (CharacterCfg): configuration of the character.
        Raises:
            FileNotFoundError: if the character's USD file is not found.
        """

        self._config: CharacterCfg = character_cfg
        self.name = character_cfg.name
        self._id = 0
        self._prim_path = self.PRIM_PREFIX + self.name + f"_{self._id}"
        self._prim: XFormPrim = None

        self._spawn_position = character_cfg.position
        self._spawn_orientation = character_cfg.orientation

        self._skeleton_tree_path: Sdf.Path = None
        self._skeleton_root_path: Sdf.Path = None

        self._animaion_interface: AnimationInterface = None
        self._skeleton_interface: SkeletonInterface = None

        self.loaded = False
        self._initialized = False
        self._scale = character_cfg.scale

        self.eye: Camera = None
        self.joint_name_index = OrderedDict()

        self.usd_path = str(RootPath.CHARACTER / self.name / "main.usd")
        if not os.path.exists(self.usd_path):
            raise FileNotFoundError(f"Character {self.name} not found")

    def load(self):
        """Load the character from the USD file, and detect the skeleton and skelroot
        prims path."""
        if self.loaded:
            return
        if prim_utils.is_prim_path_valid(self._prim_path):
            self._id = 0
            self._prim_path = self.PRIM_PREFIX + self.name + f"_{self._id}"
            while prim_utils.is_prim_path_valid(self._prim_path):
                self._id += 1
                self._prim_path = self.PRIM_PREFIX + self.name + f"_{self._id}"

        prim = stage_utils.add_reference_to_stage(self.usd_path, self._prim_path)
        UsdGeom.Xformable(prim)
        self._prim = XFormPrim(prim.GetPath().pathString)
        self.set_scale(self._scale)
        self.set_world_transform(self._spawn_position, self._spawn_orientation)

        # update the skeleton and animation prim path
        self._skeleton_tree_path = prim_utils.get_first_matching_child_prim(
            self._prim_path,
            predicate=lambda prim_path: prim_utils.get_prim_at_path(prim_path).IsA(
                UsdSkel.Skeleton
            ),
        ).GetPath()
        self._skeleton_root_path = prim_utils.get_first_matching_child_prim(
            self._prim_path,
            predicate=lambda prim_path: prim_utils.get_prim_at_path(prim_path).IsA(
                UsdSkel.Root
            ),
        ).GetPath()
        self._skeleton_interface = SkeletonInterface(self._skeleton_tree_path)

        self.initialize()
        self.loaded = True

    def remove(self):
        """Unload the character from the stage."""
        if not self.loaded:
            return
        prim_utils.delete_prim(self._prim_path)
        self.loaded = False
        self._initialized = False

    def initialize(self):
        """Initialize the character's animation control interface and perception."""
        if not self._initialized:
            self._initialize_animation()
            self._initialize_perception()
            self._initialized = True

    def _initialize_animation(self):
        """
        Initialize the animation for the character. This function creates a new
        "NewAnim" prim in the skeleton tree path and sets the animation target to
        this prim. Therefore any animation connected to the character before this
        function will not affect the character.
        """
        # create a new "NewAnim" prim neverthless it exists for controlling
        animation_path = self._skeleton_tree_path.AppendChild("NewAnim")

        self._animaion_interface = AnimationInterface(animation_path)
        self._skeleton_interface.set_animation_target(animation_path)
        self._animaion_interface.set_joints(self._skeleton_interface.get_joint_order())
        self._animaion_interface.set_local_transforms(
            self.get_rest_joint_local_transforms()
        )

        for i, name in enumerate(self.get_joint_names()):
            self.joint_name_index[name] = i

    def _initialize_perception(self):
        """
        Create the perception sensors for the character. This function makes some
        modifications to the default camera configuration to make it suitable for
        the character's eye. The camera is attached to the head of the character
        with the given position and orientation provided in the configuration.
        """
        camera_config = default_camera_cfg
        head_index = self.get_joint_name_index(self._config.head_name)
        head_path = self._skeleton_interface.get_joint_order()[head_index]
        head_prim_path = self._skeleton_tree_path.AppendPath(head_path)
        camera_config.cam_params["parent"] = head_prim_path.pathString
        camera_config.cam_params["clipping_range"] = (0.2, 50)
        camera_config.cam_params["name"] = self._prim_path.split("/")[-1] + "_eye"
        camera_config.cam_params["focal_length"] = 12.0
        # Disable semantic segmentation for performance
        camera_config.annotators["semantic_segmentation"] = False
        self.eye = Camera(camera_config)
        position = self._config.eye_position_wrt_head
        orientation = self._config.eye_orientation_wrt_head
        self.eye.set_local_pose(position, orientation)

    def step(self, pose: Iterable[Sequence[float]] = None):
        """Step the character with the given action.

        Args:
            pose (Iterable[Sequence[float]]): pose to step the character.
                The pose should be a list of joint orientations followed by
                the position and rotation of the character. Joint rotations
                and character rotations should be in quaternion format (w, x, y, z).
                Character position should be in (x, y, z) format. Defaults to None,
                which means no action is taken.
        """
        if pose is None:
            return
        assert len(pose) == len(self.get_joint_names()) + 2, (
            f"Action length is {len(pose)} but,"
            f"the character has {len(self.get_joint_names())} joints "
            "and 2 additional values for position and rotation."
        )
        self.set_joint_relative_orientations(pose[:-2])
        self.set_world_transform(pose[-2], pose[-1])

    def get_default_pose(self) -> List[Sequence[float]]:
        """The default pose for the character for user's convenience.

        Returns:
            List[Sequence[float]]: default pose of the character. The pose is a list
            of joint orientations followed by the position and rotation of the
            character. Joint rotations and character rotations are in quaternion format
            (w, x, y, z). Character position is in (x, y, z) format.
        """
        njoint = len(self.get_joint_names())
        joint_orientations = [[1, 0, 0, 0] for _ in range(njoint)]
        action = joint_orientations + [self._spawn_position, self._spawn_orientation]
        return action

    def get_joint_names(self):
        """Character's joint names.

        Returns:
            List[str]: list of joint names
        """
        return self._skeleton_interface.get_joint_names()

    def get_joint_name_index(self, name):
        """Get the index of the joint by name.

        Args:
            name (str): name of the joint
        Returns:
            int: index of the joint
        """
        return self.joint_name_index[name]

    def get_rest_joint_local_transforms(self):
        """Get the local transforms of the character's joints in the rest pose.

        Note:
            1. The rest pose is the pose of the character when it is loaded.
            2. The transform contains the position, rotation, and scale of the joint.
            For those who only need the rotation, use
            `get_rest_joint_local_orientations` instead.

        Returns:
            Vt.Matrix4dArray: list of local transforms of the joints
        """
        return self._skeleton_interface.get_rest_joint_local_transforms()

    def get_rest_joint_local_orientations(self):
        """Get the local orientations of the character's joints in the rest pose.

        Note:
            1. The rest pose is the pose of the character when it is loaded.
            2. The orientation is in quaternion format (w, x, y, z).
            3. For those who need the full transform (eg, controlling character's
            facial expressions), use `get_rest_joint_local_transforms` instead.

        Returns:
            List[Sequence[float]]: list of orientations of the joints. The orientation
            is in quaternion format (w, x, y, z).
        """
        transforms_array = self.get_rest_joint_local_transforms()
        quats = [
            Gf.Transform(transform).GetRotation().GetQuaternion()
            for transform in transforms_array
        ]
        orienations = [[quat.GetReal(), *quat.GetImaginary()[:]] for quat in quats]
        return orienations

    def get_joint_local_orientations(self, names: List[str] = None):
        """Get the local orientations of the character's joints.

        Note:
            The orientation only contains the rotation of the joint. For those who
            need the full transform (eg, controlling character's facial expressions),
            use `get_joint_local_transforms` instead.

        Args:
            names (List[str], optional): list of joint names. If not provided,
                orientations of all the joints will be returned. Defaults to None.

        Returns:
            List[Sequence[float]]: list of orientations of the joints. The orientation
            is in quaternion format (w, x, y, z). The order of the orientations is
            the same as the order of the joint names, or the order of `get_joint_names`
            if `names` is not provided.
        """
        if names is None:
            xforms = self._animaion_interface.get_local_transforms()
        else:
            xforms = self._animaion_interface.get_local_transforms_by_names(names)
        quaternions = [Gf.Transform(xform).GetRotation().GetQuat() for xform in xforms]
        orientations = [[q.GetReal(), *q.GetImaginary()] for q in quaternions]
        return orientations

    def set_joint_local_orientations(self, orientations, names: List[str] = None):
        """Set the local orientations of the character's joints.

        Note:
            1. The orientation only contains the rotation of the joint. For those who
            need the full transform (eg, controlling character's facial expressions),
            use `set_joint_local_transforms` instead.
            2. Local orientation does not include the rest pose of the character. To
            control the character's joints with respect to the rest pose, use
            `set_joint_relavitve_orientations` instead.
            3. The order of the orientations should be the same as the order of the
            joint names, or the order of `get_joint_names` if `names` is not provided.

        Args:
            orientations (List[Sequence[float]]): list of orientations of the joints.
                The orientation should be in quaternion format (w, x, y, z).
            names (List[str], optional): list of joint names. If not provided,
                orientations of all the joints will be set. Defaults to None.
        """
        if names is None:
            self._animaion_interface.set_local_orientations(orientations)
        else:
            self._animaion_interface.set_local_orientations_by_names(
                orientations, names
            )

    def set_joint_relative_orientations(self, orientations, names: List[str] = None):
        """Set the local orientations of the character's joints with respect to the
        rest pose.

        Note:
            1. This function calcualtes joints' local orientations by multiplying the
            rest pose with the given orientations. If you want to set the orientations
            directly, use `sett_joint_local_orientations` instead.
            2. The order of the orientations should be the same as the order of the
            joint names, or the order of `get_joint_names` if `names` is not provided.

        Args:
            orientations (List[Sequence[float]]): list of orientations of the joints.
                The orientation should be in quaternion format (w, x, y, z).
            names (List[str], optional): list of joint names. If not provided,
                orientations of all the joints will be set. Defaults to None.
        """
        xforms = self.get_rest_joint_local_transforms()
        if names is None:
            for i, orientation in enumerate(orientations):
                rot_mat = Gf.Matrix4d().SetRotate(Gf.Quatd(*orientation))
                xforms[i] = xforms[i] * rot_mat
        else:
            for name, orientation in zip(names, orientations, strict=False):
                xform = Gf.Matrix4d().SetRotate(Gf.Quatd(*orientation))
                index = self.get_joint_name_index(name)
                xforms[index] = xforms[index] * xform
        self._animaion_interface.set_local_transforms(xforms)

    def get_joint_relative_orientations(self, names: List[str] = None):
        """Get the local orientations of the character's joints with respect to the
        rest pose.

        Note:
            1. This function calcualtes joints' local orientations by multiplying the
            rest pose with the given orientations. If you want to get the orientations
            directly, use `get_joint_local_orientations` instead.

        Args:
            names (List[str], optional): list of joint names. If not provided,
                orientations of all the joints will be returned. Defaults to None.

        Returns:
            List[Sequence[float]]: list of orientations of the joints. The orientation
            is in quaternion format (w, x, y, z).
        """
        xforms = self.get_joint_local_transforms()
        rest_xforms = self.get_rest_joint_local_transforms()
        orientations = []
        if names is None:
            for xform, rest_xform in zip(xforms, rest_xforms, strict=True):
                rest_xform_inv = rest_xform.GetInverse()
                xform_rel = rest_xform_inv * xform
                q = Gf.Transform(xform_rel).GetRotation().GetQuat()
                orientations.append([q.GetReal(), *q.GetImaginary()])
        else:
            for name in names:
                index = self.get_joint_name_index(name)
                rest_xform_inv = rest_xforms[index].GetInverse()
                xform_rel = rest_xform_inv * xforms[index]
                q = Gf.Transform(xform_rel).GetRotation().GetQuat()
                orientations.append([q.GetReal(), *q.GetImaginary()])
        return orientations

    def set_joint_local_transforms(self, xforms, names: List[str] = None):
        """Set the local transforms of the character's joints.

        Note:
            1. The transform contains the position, rotation, and scale of the joint.
            For those who only need the rotation, use `set_joint_local_orientations`
            instead.
            2. The order of the transforms should be the same as the order of the joint
            names, or the order of `get_joint_names` if `names` is not provided.

        Args:
            xforms (List[Gf.Matrix4d]): list of local transforms of the joints
            names (List[str], optional): list of joint names. If not provided,
                transforms of all the joints will be set. Defaults to None.
        """
        if names is None:
            self._animaion_interface.set_local_transforms(xforms)
        else:
            self._animaion_interface.set_local_transforms_by_names(xforms, names)

    def get_joint_local_transforms(self, names: List[str] = None):
        """Get the local transforms of the character's joints.

        Note:
            The transform contains the position, rotation, and scale of the joint.
            For those who only need the rotation, use `get_joint_local_orientations`
            instead.

        Args:
            names (List[str], optional): list of joint names. If not provided,
                transforms of all the joints will be returned. Defaults to None.

        Returns:
            List[Gf.Matrix4d]: list of local transforms of the joints
        """
        if names is None:
            return self._animaion_interface.get_local_transforms()
        return self._animaion_interface.get_local_transforms_by_names(names)

    def set_world_transform(
        self,
        pos: Optional[Sequence[float]] = None,
        rot: Optional[Sequence[float]] = None,
    ):
        """Sets prim's pose with respect to the world's frame

        .. warning::

            This method will change (teleport) the character pose immediately to the
            indicated value

        Args:
            position (Optional[Sequence[float]], optional): position in the world frame
            of the prim. shape is (3, ). Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in
            the world frame of the prim. quaternion is scalar-first (w, x, y, z). shape
            is (4, ). Defaults to None, which means left unchanged.
        """
        self._prim.set_world_pose(pos, rot)

    def get_world_transform(self):
        """Get character's pose with respect to the world's frame

        Returns:
            Tuple[np.ndarray, np.ndarray]: first index is the position in the world
                frame (with shape (3, )). Second index is quaternion orientation (with
                shape (4, )) in the world frame
        Returns:
        """
        return self._prim.get_world_pose()

    def set_scale(self, scale: float = None):
        """Set character's scale with respect to the local frame

        Args:
            scale (float): scale to be applied to the prim's dimensions. Three
                dimensions are scaled uniformly. Defaults to None, which means left
                unchanged.
        """
        self._scale = scale
        self._prim.set_local_scale([scale, scale, scale])

    def reset(self):
        """Reset the character to the initial state."""
        if not self.loaded:
            return
        self.set_world_transform(self._spawn_position, self._spawn_orientation)
        self._animaion_interface.set_local_transforms(
            self.get_rest_joint_local_transforms()
        )
        self.set_scale(self._scale)
