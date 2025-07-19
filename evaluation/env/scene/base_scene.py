import copy
import os
import trimesh
import numpy as np
import open3d as o3d
from typing import List, Union
from math import cos, sin, pi
from yourdfpy import URDF, Link
from pathlib import Path
from utils.transform import TransformationMatrix2QuaternionXYZ
from utils.path import RootPath
import copy
from pathlib import Path
from lxml import etree

class Scene():
    """
    Provides high level functions to deal with a scene.
    NOTE: Contains collision check with agent.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._urdf_collision_path = str(
            RootPath.SCENE / self.name / "main_without_flooring.urdf"
        )
        self._urdf_visual_path = str(
            RootPath.SCENE / self.name / "main_without_ceiling.urdf"
        )
        self._urdf_path = str(
            RootPath.SCENE / self.name / "main.urdf"
        )
        self._create_urdf_without_flooring()
        self._create_urdf_without_ceiling()

        self._attachment_path = str(
            RootPath.SCENE / self.name / "attachments.json"
        )
        self._grasp_objects_positions = str(
            RootPath.SCENE / self.name / "grasp_objects_positions.json"
        )
        self._sdf_path = str(
            RootPath.SCENE / self.name / "sdf.npy"
        )
        self.grasping_poses_dir = RootPath.SCENE / self.name / "grasping_poses"
        self.urdf = URDF.load(self._urdf_path)
        self.urdf_collision = URDF.load(self._urdf_collision_path)
        self.urdf_visual = URDF.load(self._urdf_visual_path)

    @property
    def trimesh_original(self):
        self._scene = self.urdf.scene
        self._trimesh_original = self._scene.dump(concatenate=True)
        return self._simple_trimesh(self._trimesh_original)
    
    @property
    def trimesh_collision(self):
        self._scene_collision = self.urdf_collision.scene
        self._trimesh_collision = self._scene_collision.dump(concatenate=True)
        return self._simple_trimesh(self._trimesh_collision)
    
    @property
    def trimesh_visual(self):
        self._scene_visual = self.urdf_visual.scene
        self._trimesh_visual = self._scene_visual.dump(concatenate=True)
        return self._simple_trimesh(self._trimesh_visual)
    
    @property
    def trimesh_original_vis(self):
        self._scene = self.urdf.scene
        self._trimesh_original = self._scene.dump(concatenate=False)
        return self._trimesh_original
    
    @property
    def trimesh_collision_vis(self):
        self._scene_collision = self.urdf_collision.scene
        self._trimesh_collision = self._scene_collision.dump(concatenate=False)
        return self._trimesh_collision
    
    @property
    def trimesh_visual_vis(self):
        self._scene_visual = self.urdf_visual.scene
        self._trimesh_visual = self._scene_visual.dump(concatenate=False)
        return self._trimesh_visual
    
    def transform(self, matrix: Union[List, np.ndarray]):
        """
        matrix ((4, 4) float) - Homogeneous transformation matrix
        """
        self.urdf.scene.apply_transform(np.asarray(matrix))
        self.urdf_collision.scene.apply_transform(np.asarray(matrix))
        self.urdf_visual.scene.apply_transform(np.asarray(matrix))
    
    def translation(self, vector: Union[List, np.ndarray]):
        """
        vector ((3,) float) - Translation in XYZ
        """
        self.urdf.scene.apply_translation(np.asarray(vector))
        self.urdf_collision.scene.apply_translation(np.asarray(vector))
        self.urdf_visual.scene.apply_translation(np.asarray(vector))
    
    def export(self, path: str):
        self.urdf.scene.export(path)

    def export_visual(self, path: str):
        self.urdf_visual.scene.export(path)

    def export_collision(self, path: str):
        self.urdf_collision.scene.export(path)

    def update_object_position_by_transformation_matrix(
        self,  
        attach_link_name: str,
        transformation_matrix: Union[List, np.ndarray]
    ) -> None:
        """
        Change the pose of attach_link in the scene.

        Arguements:
            attach_link_name {str} -- The name of attach_link
            transformation_matrix {Union[List, np.ndarray]} -- The transformation matrix from "world" to "attach_link"
        """
        self.urdf.scene.graph.update(
            frame_to=attach_link_name, 
            frame_from="world", 
            matrix=np.asarray(transformation_matrix)
        )
        self.urdf_collision.scene.graph.update(
            frame_to=attach_link_name, 
            frame_from="world", 
            matrix=np.asarray(transformation_matrix)
        )
        self.urdf_visual.scene.graph.update(
            frame_to=attach_link_name, 
            frame_from="world", 
            matrix=np.asarray(transformation_matrix)
        )

    def return_init_state(self):
        self.urdf_collision = URDF.load(self._urdf_collision_path)
        self.urdf_visual = URDF.load(self._urdf_visual_path)
    
    def get_fixed_link_tf(self, link_name: str):
        """
        Get fixed link tf, format [x, y, z, x, y, z, w]
        """
        T = self.urdf.get_transform(frame_to=link_name, frame_from="world")
        q, xyz = TransformationMatrix2QuaternionXYZ(T)
        return [xyz[0], xyz[1], xyz[2], q[1], q[2], q[3], q[0]]
    
    def get_link_in_scene(self, link_name: str) -> trimesh.Trimesh:
        """
        Get the trimesh object in the scene frame.
        """
        link: Link = self.urdf.link_map[link_name]
        link_visuals = link.visuals[0]
        link_origin = link_visuals.origin
        link_geometry = link_visuals.geometry
        link_mesh = link_geometry.mesh
        link_filename = link_mesh.filename
        link_scale = link_mesh.scale
        link_file_path = os.path.join(
            str(RootPath.SCENE / self.name), link_filename
        )
        link_trimesh = trimesh.load(link_file_path, force="mesh")
        link_trimesh.apply_scale(link_scale)
        link_trimesh.apply_transform(
            self.urdf_visual.get_transform(
                frame_to=link_name, 
                frame_from="world"
            ) @ link_origin
        )
        return self._simple_trimesh(link_trimesh)
    
    @staticmethod
    def get_link_assert_name(urdf_file, link_name):
        tree = etree.parse(urdf_file)
        root = tree.getroot()

        for link in root.findall('link'):
            if link.get('name') == link_name:
                visual = link.find('visual')
                if visual is not None:
                    geometry = visual.find('geometry')
                    if geometry is not None:
                        mesh = geometry.find('mesh')
                        if mesh is not None:
                            file_path = mesh.get('filename')
                            return file_path.split('/')[-1].split('.')[0]
        return None
    
    def get_link(self, link_name: str) -> trimesh.Trimesh:
        """
        Get the trimesh object in the object frame.
        """
        link: Link = self.urdf.link_map[link_name]
        link_visuals = link.visuals[0]
        link_origin = link_visuals.origin
        link_geometry = link_visuals.geometry
        link_mesh = link_geometry.mesh
        link_filename = link_mesh.filename
        link_scale = link_mesh.scale
        link_file_path = os.path.join(
            str(RootPath.SCENE / self.name), link_filename
        )
        link_trimesh = trimesh.load(link_file_path, force="mesh")
        link_trimesh.apply_scale(link_scale)
        link_trimesh.apply_transform(link_origin)
        return self._simple_trimesh(link_trimesh)
    
    def _simple_trimesh(self, link_trimesh: trimesh.Trimesh) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=link_trimesh.vertices,
            faces=link_trimesh.faces,
            vertex_colors=link_trimesh.visual.to_color().vertex_colors,
            process=True,
        )
    
    def get_object_points_in_scene(
        self, 
        attach_link_name: str,
        sample_num: int=1024,
    ) -> np.ndarray:
        attach_link_trimesh = self.get_link_in_scene(attach_link_name)
        attach_link_points, _ = trimesh.sample.sample_surface(attach_link_trimesh, sample_num)
        return np.asarray(attach_link_points)
    
    def get_object_points(
        self, 
        attach_link_name: str,
        sample_num: int=1024,
    ) -> np.ndarray:
        attach_link_trimesh = self.get_link(attach_link_name)
        attach_link_points, _ = trimesh.sample.sample_surface(attach_link_trimesh, sample_num)
        return np.asarray(attach_link_points)
    
    def crop_scene_and_sample_points(
        self,
        transformation_matrix: Union[List, np.ndarray],
        sample_num: int,
        LWH: List[float],
        min_z: float = -0.01,
        sample_color: bool = False, #! when sample_color is true, there is wrong because of trimesh.
    ) -> np.ndarray:
        """
        The scene is cropped under the agent's local view and point cloud sampling is performed,
        Point clouds are relative to the world coordinate system.
        """
        slice_plane_set = {
            "plane1": [
                [-1 + LWH[0], 0, 0],
                [-1, 0, 0],
            ],
            "plane2": [
                [-1, 0, 0],
                [1, 0, 0],
            ],    
            "plane3": [
                [0, LWH[1] / 2, 0],
                [0, -1, 0],
            ],    
            "plane4": [
                [0, - LWH[1] / 2, 0],
                [0, 1, 0],
            ],    
            "plane5": [
                [0, 0, LWH[2]],
                [0, 0, -1],
            ],    
            "plane6": [
                [0, 0, min_z],
                [0, 0, 1],
            ],
        }

        scene_slice = self.trimesh_collision
        count = len(slice_plane_set.keys())
        for i in range(count):
            slice_plane = slice_plane_set[f"plane{i+1}"]
            plane_origin = np.asarray(slice_plane[0])
            plane_normal = np.asarray(slice_plane[1])
            plane_origin_homog = np.concatenate((plane_origin, np.ones(1)), axis=0)
            plane_origin_transform = np.matmul(np.asarray(transformation_matrix), plane_origin_homog)[:3]
            plane_normal_transform = np.matmul(np.asarray(transformation_matrix)[:3,:3], plane_normal)

            scene_slice = scene_slice.slice_plane(
                plane_origin_transform,
                plane_normal_transform,
            )
        
        if sample_color:    
            crop_scene_points, _, crop_scene_colors = trimesh.sample.sample_surface(
                scene_slice, sample_num, sample_color=sample_color
            )
            return crop_scene_points, crop_scene_colors
        else:
            crop_scene_points, _ = trimesh.sample.sample_surface(scene_slice, sample_num, sample_color=sample_color)
            return crop_scene_points

    def get_boundaries(self):
        self._pcd, _ = trimesh.sample.sample_surface(self.trimesh_visual, 512)
        min_boundaries = np.min(np.asarray(self._pcd), axis=0)
        max_boundaries = np.max(np.asarray(self._pcd), axis=0)
        return min_boundaries, max_boundaries
    
    def sample_pointcloud_from_trimesh_visual(self, sample_num:int, sample_color:bool=False):
        if sample_color:    
            points, _, colors = trimesh.sample.sample_surface(
                self.trimesh_visual, sample_num, sample_color=sample_color
            )
            return points, colors
        else:
            points, _ = trimesh.sample.sample_surface(
                self.trimesh_visual, sample_num, sample_color=sample_color
            )
            return points
    
    def _del_link_from_urdf(self, key:str, origin_path:str, save_path:str, isoverwrite:bool=False):
        if (not os.path.isfile(save_path)) or (isoverwrite):
            main_urdf_path = origin_path
            main_urdf = copy.deepcopy(etree.parse(source=main_urdf_path))
            root_main = main_urdf.getroot()
            floor_joints = main_urdf.xpath(f"//joint[contains(@name,{key}) and @type='fixed']")
            
            for floor_joint in floor_joints:            
                if floor_joint.find("parent").attrib["link"] == "world":
                    clild_name = floor_joint.find("child").attrib["link"]
                    if key in clild_name and "link" in clild_name:                   
                        frmat_link = "//link[@name='{}']".format(clild_name)                    
                        floor_link = main_urdf.xpath(frmat_link)[0]                    
                        root_main.remove(floor_link)
                        root_main.remove(floor_joint)        
            main_urdf.write(save_path, pretty_print=True, xml_declaration=True, encoding='utf-8')

    def _create_urdf_without_flooring(self):
        self._del_link_from_urdf('room', self._urdf_path, self._urdf_collision_path)
    
    def _create_urdf_without_ceiling(self):
        self._del_link_from_urdf('ceiling', self._urdf_collision_path, self._urdf_visual_path)
        # self._del_link_from_urdf('ceiling', self._urdf_path, self._urdf_visual_path, True)
