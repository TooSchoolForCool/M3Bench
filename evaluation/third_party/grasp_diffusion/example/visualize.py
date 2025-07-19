from typing import List, Optional, Union
import numpy as np
import open3d as o3d
import torch
import sys
from os import path, read
from os.path import join

def visualize_point_cloud(
    point_cloud: Union[np.ndarray, o3d.geometry.PointCloud], 
    colors: Optional[np.ndarray] = None,
    vis_frame: bool=True,
    frame_size: float=0.5,
    frame_origin: List[float]=[0, 0, 0],
):
    if isinstance(point_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    elif isinstance(point_cloud, o3d.geometry.PointCloud):
        pcd = point_cloud
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        raise NotImplementedError
    
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_origin)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(left=600)
    vis.add_geometry(pcd)
    if vis_frame:
        vis.add_geometry(frame)
    
    # render_option = vis.get_render_option()
    # render_option.load_from_json(join(path.dirname(path.abspath(__file__)), "point_cloud_render_option.json"))
    
    vis.run()