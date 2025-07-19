from __future__ import annotations

import json

import numpy as np
import omni.isaac.core.utils.prims as prims_utils


class NumpyJsonEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays.

    Converts NumPy float32 and arrays to their JSON serializable equivalents.
    """

    def default(self, o):
        if isinstance(o, np.float32):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def deletable_prim_check(prim_path: str) -> bool:
    # 1. Check if a path has a valid USD prim
    if not prims_utils.is_prim_path_valid(prim_path):
        return False
    # 2. Checks whether a prim can be deleted or not from USD stage.
    if prims_utils.is_prim_no_delete(prim_path):
        return False
    # 3. Check if any of the prims ancestors were brought in as a reference
    # if this prim is an usd file loaded as reference under that path or
    # created during task running, return True
    if prims_utils.is_prim_ancestral(prim_path):
        return False
    # 4. /Render prims cause segmentation fault if removed.
    if prim_path.startswith("/Render"):
        return False
    return True
