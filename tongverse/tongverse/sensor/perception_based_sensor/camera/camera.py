from __future__ import annotations

from typing import Optional, Sequence, Tuple

import omni.replicator.core as rep
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import delete_prim
from pxr import UsdGeom
from torch import tensor

from tongverse.sensor.perception_based_sensor.camera.annotator_wrapper import (
    get_annotation_wrapper,
)
from tongverse.sensor.perception_based_sensor.camera.camera_cfg import (
    CameraCfg,
    default_camera_cfg,
)
from tongverse.utils import deletable_prim_check


class Camera:
    """
    This class represents a camera object that performs the following tasks:

    1. Creates a camera prim based on the specified camera parameters.
    2. Creates a render product and attaches the created camera to it.
       A RenderProduct describes images or other file-like artifacts produced
       by a render,such as RGB (LdrColor), normals, depth, etc.
    3. Adds annotators to the render product based on annotator configuration.
       Annotators provide ground truth annotations corresponding to the rendered scene.
    4. Provides a save function to save data for the current simulation frame.

    NOTE:
        1. The first few frames of the camera output may be blank immediately after
        the environment is initialized.This occurs because the simulator may require a
        few steps to load all material textures properly.
        2. DO NOT set rotation of camera if it has a look-at target specified in
        cam_params configuration.
    """

    def __init__(self, camera_cfg: CameraCfg = default_camera_cfg):
        """
        Initialize the Camera object with the given camera configuration.

        Args:
            camera_cfg (CameraCfg, optional): The camera configuration. If None,
            default configuration is used.
        """
        cam_params = camera_cfg.cam_params
        # 1. Create camera
        self._camera = rep.create.camera(**cam_params)
        self._name = cam_params["name"]
        prefix = "/Replicator" if cam_params["parent"] is None else cam_params["parent"]
        self._xform_prim = XFormPrim(f"{prefix}/{cam_params['name']}_Xform")
        self._add_xformops()
        # 2. Create render product and attach camera to render product
        self._render_product = rep.create.render_product(
            self._camera, resolution=camera_cfg.resolution
        )
        # 3. Add annotator to render product
        self._annotators = {}
        annotators = camera_cfg.annotators
        self._add_annotator_to_rp(
            self._render_product, annotators, camera_cfg.output_dir
        )

    def remove(self):
        if deletable_prim_check(self.prim_path):
            delete_prim(self.prim_path)

    def save(self) -> None:
        """
        Manually saves data for the current simulation frame to the designated output.

        This method iterates through all annotators associated with the render product
        and saves data using the camera name and provided output directory as a
        reference.
        """
        for anno in self._annotators.values():
            anno.save(self._name)

    def stream(
        self, rgb: bool = True, segmentation: bool = True
    ) -> Tuple[Optional[bytes], Optional[bytes]]:
        """
        Provides data buffer for output streaming used for web applications.

        Args:
            rgb (bool, optional): Whether to stream RGB data. Defaults to True.
            segmentation (bool, optional): Whether to stream segmentation data.
            Defaults to True.

        Returns:
            Tuple[Optional[bytes],Optional[bytes]]: A tuple containing RGB and
            segmentation streams if available.

        """
        rgb_stream, seg_stream = None, None
        if rgb and self._annotators.get("rgb"):
            rgb_stream = self._annotators["rgb"].stream()
        if segmentation and self._annotators.get("semantic_segmentation"):
            seg_stream = self._annotators["semantic_segmentation"].stream()
        return rgb_stream, seg_stream

    def get_render_product_cam_params(self) -> dict:
        """
        Returns the camera details for the camera associated with the render product to
        which the annotator is attached.

        Raises:
            KeyError: If the CameraParams annotator is not attached to the camera.

        Returns:
            dict: A dictionary containing camera details, including aperture offset,
            focal length,focus distance, f-stop, resolution, and other parameters
            associated with the camera.
        """
        if not self._annotators.get("CameraParams"):
            raise KeyError("CameraParams annotator is not attached")
        return self._annotators.get("CameraParams").annotator.get_data()

    def get_annotators(self) -> dict:
        """
        Retrieve annotators associated with the camera.

        Returns:
            dict: A dictionary where keys are annotator names and values are annotator
            objects. Annotators provide ground truth annotations corresponding to the
            rendered scene.
        """
        return self._annotators

    def get_world_pose(self) -> Tuple[tensor, tensor]:
        """
        Retrieves the world pose (position and orientation) of the camera xform prim

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the positions and orientations
            of the object.
        """
        pos, orient = self.prim.get_world_pose()
        return pos, orient

    def get_local_pose(self) -> Tuple[tensor, tensor]:
        """
        Retrieves the local pose (position and orientation) of the camera relative
        to its parent.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the translation and
            orientation tensors of the camera relative to its parent.
        """
        translation, orient = self.prim.get_local_pose()
        return translation, orient

    def set_world_pose(
        self,
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Sets the world pose of the camera xform prim.

        Args:
            position (Optional[Sequence[float]]): The position to set.
            orientation (Optional[Sequence[float]]): The orientation to set.

        NOTE: if the camera has a look-at-target, avoid setting its orientation.
        """
        if position is not None:
            self.prim.set_world_pose(position=position)

        if orientation is not None:
            self.prim.set_world_pose(orientation=orientation)

    def set_local_pose(
        self,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Sets the local pose of the camera xform prim.

        Args:
            translation (Optional[Sequence[float]]): The translation to set.
            orientation (Optional[Sequence[float]]): The orientation to set.
        """
        if translation is not None:
            self.prim.set_local_pose(translation=translation)

        if orientation is not None:
            self.prim.set_local_pose(orientation=orientation)

    @property
    def name(self) -> str:
        """
        Get the camera name.
        """
        return self._name

    @property
    def prim(self) -> XFormPrim:
        """
        Get the XFormPrim associated with the camera.
        """
        return self._xform_prim

    @property
    def prim_path(self):
        """
        Get the prim path associated with the XFormPrim of the camera.
        """
        return self._xform_prim.prim_path

    # =========internal helper===============
    def _add_annotator_to_rp(
        self, render_product, annotators: dict, image_output: str
    ) -> None:
        """
        Internal method to add annotator to render product.

        Args:
            render_product: Render product to which annotators will be added.
            annotators (dict): Dictionary of annotators.
            image_output (str): Directory path for image output.
        """
        for anno_name, is_used in annotators.items():
            if is_used:
                self._annotators[anno_name] = get_annotation_wrapper(
                    anno_name, render_product, image_output
                )

    def _add_xformops(self) -> None:
        """
        Internal method to add transform operations to the camera's XFormPrim.
        """
        xformable = UsdGeom.Xformable(self.prim.prim)
        xformable.AddXformOp(
            UsdGeom.XformOp.TypeRotateXYZ, UsdGeom.XformOp.PrecisionDouble, ""
        )
