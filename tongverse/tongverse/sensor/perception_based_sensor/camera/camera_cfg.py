from __future__ import annotations

import dataclasses
from typing import Tuple, Union

from tongverse.utils.constant import RootPath


class CameraCfg:
    @dataclasses.dataclass
    class CamParams:
        """
        References:
        *  https://docs.omniverse.nvidia.com/py/replicator/1.10.10/source/extensions/
            omni.replicator.core/docs/API.html#cameras
        """

        look_at: Union[str, Tuple[float, float, float]] = None
        """
        Look-at target, specified either as a prim path, or world coordinates.
        NOTE: Rotation should be None when look_at_target is specified
        """
        look_at_up_axis: Tuple[float] = None
        focal_length: float = 24.0
        focus_distance: float = 400.0
        f_stop: float = 0.0
        horizontal_aperture: float = 20.955
        horizontal_aperture_offset: float = 0.0
        vertical_aperture_offset: float = 0.0
        clipping_range: Tuple[float, float] = (1.0, 1000000.0)
        projection_type: str = "pinhole"
        fisheye_nominal_width: float = 1936.0
        fisheye_nominal_height: float = 1216.0
        fisheye_optical_centre_x: float = 970.94244
        fisheye_optical_centre_y: float = 600.37482
        fisheye_max_fov: float = 200.0
        fisheye_polynomial_a: float = 0.0
        fisheye_polynomial_b: float = 0.00245
        fisheye_polynomial_c: float = 0.0
        fisheye_polynomial_d: float = 0.0
        fisheye_polynomial_e: float = 0.0
        fisheye_polynomial_f: float = 0.0
        fisheye_p0: float = -0.00037
        fisheye_p1: float = -0.00074
        fisheye_s0: float = -0.00058
        fisheye_s1: float = -0.00022
        fisheye_s2: float = 0.00019
        fisheye_s3: float = -0.0002
        count: int = 1
        parent: str = None
        """
        The camera will be created as a child of this parent path
        """
        name: str = "camera"

    @dataclasses.dataclass
    class AnnotatorCfg:
        """
        References:
        *  https://docs.omniverse.nvidia.com/py/replicator/1.10.10/source/extensions/
        omni.replicator.core/docs/API.html#annotators
        *  https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/
        annotators_details.html
        """

        rgb: bool = True
        """
        The LdrColor or rgb annotator produces the low dynamic range output image as
        an array of type np.uint8 with shape (width, height, 4), where the four
        channels correspond to R,G,B,A.
        """
        semantic_segmentation: bool = True
        """
        Outputs semantic segmentation of each entity in the camera’s viewport that has
        semantic labels.
        """
        instance_id_segmentation: bool = False
        """
        Outputs instance id segmentation of each entity in the camera’s viewport.
        The instance id is unique for each prim in the scene with different paths.
        """
        instance_segmentation: bool = False
        """
        Outputs instance segmentation of each entity in the camera’s viewport.
        The main difference between instance id segmentation and instance segmentation
        are that instance segmentation annotator goes down the hierarchy to the lowest
        level prim which has semantic labels, which instance id segmentation always goes
        down to the leaf prim
        """
        bounding_box_2d_loose: bool = False
        """
        Outputs a “loose” 2d bounding box of each entity with semantics in the
        camera’s field of view
        """
        bounding_box_2d_tight: bool = False
        """
        Outputs a “tight” 2d bounding box of each entity with semantics in the
        camera’s viewport.
        """
        bounding_box_3d: bool = False
        """
        Outputs 3D bounding box of each entity with semantics in the camera’s viewport.
        """
        distance_to_camera: bool = False
        """
        Outputs a depth map from objects to camera positions.
        The distance_to_camera annotator produces a 2d array of types np.float32
        with 1 channel.
        """
        distance_to_image_plane: bool = False
        """"
        Outputs a depth map from objects to image plane of the camera.
        The distance_to_image_plane annotator produces a 2d array of types np.float32
        with 1 channel
        """
        pointcloud: bool = False
        """
        Outputs a 2D array of shape (N, 3) representing the points sampled on
        the surface of the prims in the viewport, where N is the number of point
        """
        normals: bool = False
        """
        The normals annotator produces an array of type np.float32 with
        shape (height, width, 4).
        The first three channels correspond to (x, y, z). The fourth channel is unused.
        """
        skeleton_data: bool = False
        """
        The skeleton data annotator outputs pose information about the skeletons
        in the scene view.
        """
        MotionVectors: bool = False  # pylint:disable=C0103
        """
        Outputs a 2D array of motion vectors representing the relative motion of
        a pixel in the camera’s viewport between frames.
        """
        CameraParams: bool = True  # pylint:disable=C0103
        """
        The Camera Parameters annotator returns the camera details for the camera
        corresponding to the render product to which the annotator is attached.
        """

    def __init__(self):
        self.output_dir = str(RootPath.IMG / "Perception_Cam")
        self.resolution: Tuple[float, float] = (1024, 1024)
        self.cam_params = dataclasses.asdict(self.CamParams())
        self.annotators = dataclasses.asdict(self.AnnotatorCfg())


default_camera_cfg = CameraCfg()
