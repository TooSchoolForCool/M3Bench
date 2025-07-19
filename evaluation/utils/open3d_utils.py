from typing import List, Optional, Union
import numpy as np
import open3d as o3d
import torch
import utils.pointcloud_utils as pcu

import sys
from os import path, read
from os.path import join

def read_point_cloud(pc_file) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(pc_file)
    return pcd

def read_mesh(mesh_file) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    return mesh

def save_point_cloud(pcd: o3d.geometry.PointCloud, save_path: str) -> None:
    o3d.io.write_point_cloud(save_path, pcd)

def save_mesh(mesh: o3d.geometry.TriangleMesh, save_path: str) -> None:
    o3d.io.write_triangle_mesh(save_path, mesh)

def convert_numpy_to_point_cloud(points : np.ndarray, normal: Optional[np.ndarray] = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)
        pcd.normalize_normals()
    return pcd

def set_pcd_color(pcd : o3d.geometry.PointCloud, colors : np.ndarray):
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def save_numpy_to_ply(points, save_path, normal=None):
    """[summary]

    Args:
            points (Np, 3): [points]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)
        pcd.normalize_normals()
    o3d.io.write_point_cloud(save_path, pcd)

def generate_mesh_from_points(points, normal, method="BPA"):
    """[summary]
    
    Args:
            points: np.ndarray(number of points, 3)

            methods: ["BPA" or "poisson"]
    
    Reference:
    
    https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba
    
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normal)
    if method == "BPA":
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        
        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
        
        dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)
        
        dec_mesh.remove_degenerate_triangles()
        dec_mesh.remove_duplicated_triangles()
        dec_mesh.remove_duplicated_vertices()
        dec_mesh.remove_non_manifold_edges()
        mesh = dec_mesh
    elif method == "poisson":
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        mesh = poisson_mesh
    
    return mesh

def sampling_points_from_mesh(mesh_file, number_of_points=2048, method="uniform"):
    """
    Reference: 
    http://www.open3d.org/docs/release/tutorial/geometry/mesh.html
    """
    mesh = read_mesh(mesh_file)
    if method=="uniform":
        pcd = mesh.sample_points_uniformly(number_of_points)
    elif method=="poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points, init_factor=5)
    elif method=="uniform-poisson":
        pcd = mesh.sample_points_uniformly(number_of_points*5)
        pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points, pcl=pcd)
    
    return pcd

def plane_fitting_from_points(points : np.ndarray):
    """get plane from points

    Args:
            points (np.ndarray): [(n, 3) target points]

    Returns:
            plane_model: a, b, c, d of equation {ax+by+cz=d}
            inliers: index of inlier points
    """
    assert len(points) > 3, "There is at most 3 points to fitting plane but got {}".format(len(points))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
    return plane_model, inliers
    
def slicing_by_plane(points : np.ndarray, offset=0.5):
    # get mean of points
    com = np.mean(points, axis=0)
    shifted_points = points - com
    
    # get random normal axis of slicing plane
    normal_axis = np.random.rand(3)
    
    # calculate dot product of each points
    dotproduct = np.dot(shifted_points, normal_axis)
    dotproduct /= np.max(dotproduct)

    # get partial index of sliced points
    return dotproduct > offset

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

def visualize_geometry(geometry_list):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(left=600)
    for geo in geometry_list:
        vis.add_geometry(geo)
    
    # render_option = vis.get_render_option()
    # render_option.load_from_json(join(path.dirname( path.abspath(__file__) ), "point_cloud_render_option.json"))
    
    vis.run()


def normalize_point_cloud(points : np.ndarray) -> np.ndarray:
    assert points.shape[0] > 0
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_value = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / max_value
    
    return points 


def hidden_point_removal(pcd, camera_location, radius):
    re = pcd.hidden_point_removal(camera_location, radius)
    

def down_sample_points(points: np.ndarray, num_sample, method='random') -> np.ndarray:
    assert len(points) >= num_sample

    if method == 'poisson':
        target_idx = pcu.downsample_point_cloud_poisson_disk(points, num_samples=num_sample)[:num_sample]
        if len(target_idx) < num_sample:
            target_idx = np.r_[target_idx, np.setdiff1d(np.arange(len(points)), target_idx)[:num_sample-len(target_idx)]]
    
    elif method == 'random':
        target_idx = np.random.choice(range(len(points)), num_sample)
    
    elif method == 'fps':
        target_idx = farthest_point_sample(points, num_sample)
    
    else:
        raise NotImplementedError
    
    return target_idx

def farthest_point_sample(points, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = torch.from_numpy(points).unsqueeze(0)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids[0]


class PointCloudVisualizer:
    """Reference:
    
    http://www.open3d.org/docs/0.9.0/python_api/open3d.visualization.Visualizer.html#open3d.visualization.Visualizer
    
    """

    def get_custom_pcd(self) -> o3d.geometry.PointCloud:
        """TODO: if there is preprocess for cloud file
        
        # using self.object_idx for get current object
        point_cloud_file = self.object_file_list[self.object_idx]
        
        # example for npy files
        pcd = o3d.geometry.PointCloud()
        xyz = np.load(point_cloud_file)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        return pcd
        """
        raise NotImplementedError
    
    def get_custom_pcd_color(self) -> o3d.utility.Vector3dVector:
        """TODO: if there is additional color

        current_pcd_color = self.color_list[self.object_idx]
        return o3d.utility.Vector3dVector(current_pcd_color)
        
        
        Raises:
                NotImplementedError: [description]

        Returns:
                [o3d.utility.Vector3dVector()]: [description]
        """
        
        raise NotImplementedError
    
    def __init__(self, point_cloud_file_list, custom_pcd=False, custom_color=False):
        self.object_file_list = point_cloud_file_list
        self.custom_pcd = custom_pcd
        self.custom_color = custom_color
        
        self.object_idx = -1
        self.object_num = len(self.object_file_list)
        self.current_pcd = self.get_next_pcd()
        
        # initialize visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(left=600) # margin
        
        # # add current PointCloud to window
        self.vis.add_geometry(self.current_pcd)
        
        # load render option from file : RenderOption
        self.render_option = self.vis.get_render_option()
        self.render_option.load_from_json(join(path.dirname( path.abspath(__file__) ), "o3d_render_options.json"))
        
        # register callback function 
        self.vis.register_animation_callback(self.rotate_view)
        self.vis.register_key_callback(ord("N"), self.update_pcd)

        # activate window
        self.vis.run()
    
    def get_next_pcd(self):
        self.object_idx += 1
        if not self.object_idx < self.object_num:
            print("No more Objects ==> Exit Program!!")
            exit()
        
        # next object file from list        
        point_cloud_file = self.object_file_list[self.object_idx]
        print("Get Next Object: {}".format(point_cloud_file))
        # create PointCloud
        if self.custom_pcd:
            pcd = self.get_custom_pcd()
        else:
            supported_format = [".xyz", ".ply", ".pcd", ".pts", ".xzyrgb", ".xyzn"] # http://www.open3d.org/docs/0.6.0/tutorial/Basic/file_io.html?highlight=read_point_cloud#point-cloud
            
            if path.splitext(point_cloud_file)[-1] in supported_format:
                pcd = o3d.io.read_point_cloud(point_cloud_file)
            
            else:
                assert False, "Not Supported Type {}".format(path.split(point_cloud_file)[-1])
        
        # if there is custom color
        if self.custom_color:
            pcd.colors = self.get_custom_pcd_color()

        return pcd
    
    #FIXME
    def get_previous_pcd(self):
        assert False
        return None

        
    def update_pcd(self, vis):
        vis.clear_geometries()
        self.current_pcd = self.get_next_pcd()
        vis.add_geometry(self.current_pcd)
        vis.reset_view_point(True)
        
        return False
    
    def rotate_view(self, vis):
        # ctr = vis.get_view_control()
        # ctr.rotate(5.0, 0.0)
        return False

if __name__=="__main__":
    
    whole_point_cloud = []

    PointCloudVisualizer(whole_point_cloud)
    
    
        