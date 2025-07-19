import os
import json
from pathlib import Path
from typing import List, Union
import trimesh
from trimesh import Trimesh
import numpy as np
import open3d as o3d
from yourdfpy import URDF, Link
from transform import EulerAnglesXYZ2TransformationMatrix

class RootPath:
    SCENE_BUILDER = Path("/home/ysx/0_WorkSpace/11_VKC/src/vkc_deps/scene_builder")
    SCENE = SCENE_BUILDER / "scene"

class Scene():
    """
    Provides high level functions to deal with a scene.
    NOTE: Contains collision check with agent.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._urdf_path = str(
            RootPath.SCENE / self.name / "main.urdf"
        )
        print(self._urdf_path)
        self._attachment_path = str(
            RootPath.SCENE / self.name / "attachments.json"
        )
        self._grasp_object_file_path = str(
            RootPath.SCENE / self.name / "grasp_objects.json"
        )
        self._grasp_objects_positions = str(
            RootPath.SCENE / self.name / "grasp_objects_positions.json"
        )
        self.grasping_poses_dir = RootPath.SCENE / self.name / "grasping_poses"
        self.urdf = URDF.load(self._urdf_path)
        # self.scene = self.urdf.scene
        # self.trimesh = self.scene.dump(concatenate=True)
    
    @property
    def trimesh(self):
        self._scene = self.urdf.scene
        self._trimesh = self._scene.dump(concatenate=True)
        return self._trimesh
    
    def update_object_position(
        self, 
        link_name: str, 
        joint_name: str,
        joint_value_id: int
    ):
        with open(self._grasp_objects_positions, "r") as f:
            item_json:json = f.read()
            item_dict:dict = json.loads(item_json)
            joint_value = item_dict[link_name][joint_name][str(joint_value_id)]

            origin = self.urdf.joint_map[joint_name].origin
            origin[0, 3] = joint_value[0]
            origin[1, 3] = joint_value[1]
            origin[2, 3] = joint_value[2]
            self.urdf.joint_map[joint_name].origin = origin
            self.urdf._scene = self.urdf._create_scene(
                use_collision_geometry=False,
                load_geometry=True,
                force_mesh=False,
                force_single_geometry_per_link=True,
            )
    
    def update_object_position_by_joint_value(
        self,  
        joint_name: str,
        joint_value: dict
    ):
        origin = EulerAnglesXYZ2TransformationMatrix(joint_value["rpy"], joint_value["xyz"])
        self.urdf.joint_map[joint_name].origin = origin
        self.urdf._scene = self.urdf._create_scene(
            use_collision_geometry=False,
            load_geometry=True,
            force_mesh=False,
            force_single_geometry_per_link=True,
        )
    
    def return_init_state(self):
        self.urdf = URDF.load(self._urdf_path)
    
    def get_link(self, link_name: str) -> Trimesh:
        """
        Get the trimesh object for the specified link in the scene.
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
        return link_trimesh
    
    def get_boundaries(self):
        self._pcd, _ = trimesh.sample.sample_surface(self.trimesh_visual, 512)
        min_boundaries = np.min(np.asarray(self._pcd), axis=0)
        max_boundaries = np.max(np.asarray(self._pcd), axis=0)
        return min_boundaries, max_boundaries
    
    def get_grasp_object(self) -> List[str]:
        """
        Return the object link, which could be grasped by gripper.
        """
        with open(self._grasp_object_file_path, "r") as f:
            item_json:json = f.read()
            item_dict:dict = json.loads(item_json)
            return item_dict
