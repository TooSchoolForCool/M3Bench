from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import omni.replicator.core as rep
from PIL import Image

from tongverse.utils import NumpyJsonEncoder


class DataSaver:
    @staticmethod
    def save_data_rgb(data: np.ndarray, file_stem: str, output_path: Path) -> None:
        """Save RGB image data to a PNG file."""
        buff = np.frombuffer(data, dtype=np.uint8)
        if len(buff) > 0:
            rgb_image_data = buff.reshape(*data.shape, -1)
            Image.fromarray(rgb_image_data, "RGBA").save(
                output_path / f"{file_stem}.png"
            )

    @staticmethod
    def save_data_seg(data: Dict[str, Any], file_stem: str, output_path: Path) -> None:
        """Save segmentation data along with JSON metadata."""
        DataSaver.save_data_rgb(data["data"], file_stem, output_path)
        DataSaver.save_data_json(data["info"], f"{file_stem}.info", output_path)

    @staticmethod
    def save_data_json(data: Dict[str, Any], file_stem: str, output_path: Path) -> None:
        """Save data to a JSON file."""
        with (output_path / f"{file_stem}.json").open(
            "w", encoding="utf-8"
        ) as json_file:
            json.dump(data, json_file, cls=NumpyJsonEncoder, indent=2)

    @staticmethod
    def save_data_npz(data: Any, file_stem: str, output_path: Path) -> None:
        """Save data to a NumPy NPZ file."""
        np.savez(output_path / f"{file_stem}.npz", data)


class AnnotatorWrapper:
    """
    AnnotatorWrapper creates a wrapper for annotators, providing functionality to
    save data corresponding to the current simulation frame. If the annotator is
    for RGB or segmentation data, it also offers methods to stream out the data.

    Attributes:
        anno_name (str): The name of the annotator.
        render_product (object): The render product associated with the annotator.
        data_root (str): The root directory for saving data.

    Methods:
        __init__(name: str, render_product: object, data_root: str):
            Initializes the AnnotatorWrapper with the given name, render product,
            and data root.

        save(cam_name: str) -> None:
            Saves data for the current simulation frame under the specified camera name.

        _attach() -> None:
            Internal method to attach the annotator to the render product.

        _save(data: Any, file_stem: str, output_path: Path) -> None:
            Internal method to save data associated with the annotator.

        _stream() -> Optional[bytes]:
            Internal method to stream out data associated with the annotator.

        stream() -> Optional[bytes]:
            Streams out data for the current simulation frame.
    """

    def __init__(self, anno_name: str, render_product: object, data_root: str) -> None:
        self.anno_name = anno_name
        self.data_root = data_root
        self.render_product = render_product
        self._attach()

    def _attach(self) -> None:
        """
        Internal method to attach the annotator to the render product.
        """
        self.annotator = rep.annotators.get(self.anno_name).attach(
            [self.render_product]
        )

    def _save(self, data: Any, file_stem: str, output_path: Path) -> None:
        """
        Internal method to save data associated with the annotator.
        """
        return DataSaver.save_data_json(data, file_stem, output_path)

    def save(self, cam_name: str) -> None:
        """Save data for the current simulation frame."""
        output_path = Path(f"{self.data_root}/{cam_name}")
        output_path.mkdir(exist_ok=True, parents=True)

        data = self.annotator.get_data()
        ts = round(time.time() * 1000)
        file_stem = f"{ts}-{self.anno_name}"
        return self._save(data, file_stem, output_path)

    def _stream(self) -> Optional[bytes]:
        """Internal method to stream out data associated with the annotator."""

    def stream(self) -> Optional[bytes]:
        """Stream data for the current simulation frame."""
        return self._stream()


class RGBAnnotatorWrapper(AnnotatorWrapper):
    def _save(self, data: np.ndarray, file_stem: str, output_path: Path) -> None:
        """Save RGB data."""
        return DataSaver.save_data_rgb(data, file_stem, output_path)

    def _stream(self) -> Optional[bytes]:
        """Stream RGB data."""
        data = self.annotator.get_data()
        if data is not None and len(data) > 0:
            # pylint: disable=no-member
            rgb_image_data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            # pylint: disable=no-member
            rgb_image_array = cv2.resize(
                rgb_image_data,
                (rgb_image_data.shape[1], int(rgb_image_data.shape[1] / 16 * 10)),
            )
            # pylint: disable=no-member
            _, buffer = cv2.imencode(".png", rgb_image_array)
            return buffer
        return None


class BBoxAnnotatorWrapper(AnnotatorWrapper):
    def _attach(self):
        self.annotator = rep.annotators.get(
            self.anno_name, init_params={"semanticTypes": ["class"]}
        ).attach([self.render_product])


class SegmentationAnnotatorWrapper(AnnotatorWrapper):
    def _attach(self):
        """Attach segmentation annotator."""
        self.annotator = rep.annotators.get(
            self.anno_name, init_params={"colorize": True}
        ).attach([self.render_product])

    def _save(self, data: Dict[str, Any], file_stem: str, output_path: Path) -> None:
        """Save segmentation data."""
        return DataSaver.save_data_seg(data, file_stem, output_path)

    def _stream(self) -> Optional[bytes]:
        """Stream segmentation data."""
        sem_data = self.annotator.get_data()
        if sem_data is not None and len(sem_data) > 0:
            sem_image_data = np.frombuffer(sem_data["data"], dtype=np.uint8).reshape(
                *sem_data["data"].shape, -1
            )
            # pylint: disable=no-member
            sem_image_array = cv2.resize(
                sem_image_data,
                (sem_image_data.shape[1], int(sem_image_data.shape[1] / 16 * 10)),
            )
            # pylint: disable=no-member
            _, buffer = cv2.imencode(".png", sem_image_array)

            return buffer
        return None


class NpzAnnotatorWrapper(AnnotatorWrapper):
    def _save(self, data: Any, file_stem: str, output_path: Path) -> None:
        """Save NPZ data."""
        return DataSaver.save_data_npz(data, file_stem, output_path)


def get_annotation_wrapper(
    anno_name: str, render_product: object, data_root: str
) -> AnnotatorWrapper:
    """Factory function to get the appropriate annotator wrapper."""
    return ANNOTATOR_WRAPPER[anno_name](anno_name, render_product, data_root)


ANNOTATOR_WRAPPER = {
    "rgb": RGBAnnotatorWrapper,
    "bounding_box_2d_loose": BBoxAnnotatorWrapper,
    "bounding_box_2d_tight": BBoxAnnotatorWrapper,
    "bounding_box_3d": BBoxAnnotatorWrapper,
    "semantic_segmentation": SegmentationAnnotatorWrapper,
    "instance_id_segmentation": SegmentationAnnotatorWrapper,
    "instance_segmentation": SegmentationAnnotatorWrapper,
    "pointcloud": NpzAnnotatorWrapper,
    "CameraParams": AnnotatorWrapper,
    "distance_to_camera": AnnotatorWrapper,
    "distance_to_image_plane": AnnotatorWrapper,
    "MotionVectors": AnnotatorWrapper,
    "normals": AnnotatorWrapper,
    "occlusion": AnnotatorWrapper,
    "skeleton_data": AnnotatorWrapper,
}
