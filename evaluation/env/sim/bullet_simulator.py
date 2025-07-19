import time
from pathlib import Path
from typing import List
import numpy as np
import pybullet as p
from geometrout.primitive import Cuboid, Sphere, Cylinder
from geometrout.transform import SE3
from pyquaternion import Quaternion
import math
from cprint import *
from urchin import URDF
from env.agent.mec_kinova import MecKinova
from utils.transform import transform_pointcloud_torch

class Bullet:
    def __init__(self, gui=False):
        """
        :param gui: Whether to use a gui to visualize the environment.
            Only one gui instance allowed
        """
        self.use_gui = gui
        if self.use_gui:
            self.clid = p.connect(p.GUI)
        else:
            self.clid = p.connect(p.DIRECT)
        self.robots = {}
        self.obstacle_ids = []

    def __del__(self):
        """
        Disconnects the client on destruction
        """
        p.disconnect(self.clid)

    def set_camera_position(self, yaw, pitch, distance, target):
        p.resetDebugVisualizerCamera(
            distance, yaw, pitch, target, physicsClientId=self.clid
        )

    def get_camera_position(self):
        params = p.getDebugVisualizerCamera(physicsClientId=self.clid)
        return {
            "yaw": params[8],
            "pitch": params[9],
            "distance": params[10],
            "target": params[11],
        }

    def get_depth_and_segmentation_images(
        self,
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        near,
        far,
        camera_T_world,
        scale=True,
    ):
        projection_matrix = (
            2.0 * fx / width,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * fy / height,
            0.0,
            0.0,
            1.0 - 2.0 * cx / width,
            2.0 * cy / height - 1.0,
            (far + near) / (near - far),
            -1.0,
            0.0,
            0.0,
            2.0 * far * near / (near - far),
            0.0,
        )
        view_matrix = camera_T_world.matrix.T.reshape(16)
        _, _, _, depth, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.clid,
        )
        if scale:
            depth = far * near / (far - (far - near) * depth)
        return depth, seg

    def get_pointcloud_from_camera(
        self,
        camera_T_world,
        width=640,
        height=480,
        fx=616.36529541,
        fy=616.20294189,
        cx=310.25881958,
        cy=310.25881958,
        near=0.01,
        far=10,
        remove_robot=None,
        keep_robot=None,
        finite_depth=True,
    ):
        assert not (keep_robot is not None and remove_robot is not None)
        depth_image, segmentation = self.get_depth_and_segmentation_images(
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near,
            far * 2,
            camera_T_world,
        )
        # Remove all points that are too far away
        depth_image[depth_image > far] = 0.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        if remove_robot is not None:
            depth_image[segmentation == remove_robot.id] = 0.0
        elif keep_robot is not None:
            depth_image[segmentation != keep_robot.id] = 0.0
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width))
        image_points = np.stack((x, y, ones), axis=2).reshape(width * height, 3).T
        backprojected = np.linalg.inv(K) @ image_points
        pc = np.multiply(
            np.tile(depth_image.reshape(1, width * height), (3, 1)), backprojected
        ).T
        if finite_depth:
            pc = pc[np.isfinite(pc[:, 0]), :]
        capture_camera = camera_T_world.inverse @ SE3(xyz=[0, 0, 0], quat=[0, 1, 0, 0])
        pc = pc[~np.all(pc == 0, axis=1)]
        transform_pointcloud_torch(pc, capture_camera.matrix, in_place=True)
        return pc

    def load_robot(self, robot_type, **kwargs):
        """
        Generic function to load a robot.
        """
        if robot_type == MecKinova:
            robot = BulletMecKinova(self.clid, **kwargs)
        self.robots[robot.id] = robot
        return robot

    def in_collision(self, robot, radius=0.0, check_self=False):
        return robot.in_collision(self.obstacle_ids, radius, check_self)

    def load_mesh(self, visual_mesh_path, collision_mesh_path=None, color=None):
        if collision_mesh_path is None:
            collision_mesh_path = visual_mesh_path
        if color is None:
            color = [0.85882353, 0.14117647, 0.60392157, 1]
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=visual_mesh_path,
            rgbaColor=color,
            physicsClientId=self.clid,
        )
        collision_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=collision_mesh_path,
            physicsClientId=self.clid,
        )
        obstacle_id = p.createMultiBody(
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            baseVisualShapeIndex=visual_id,
            baseCollisionShapeIndex=collision_id,
            physicsClientId=self.clid,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_cuboid(self, cuboid, color=None, visual_only=False):
        assert isinstance(cuboid, Cuboid)
        if color is None:
            color = [0.85882353, 0.14117647, 0.60392157, 1]
        assert not cuboid.is_zero_volume(), "Cannot load zero volume cuboid"
        kwargs = {}
        if self.use_gui:
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=cuboid.half_extents,
                rgbaColor=color,
                physicsClientId=self.clid,
            )
            kwargs["baseVisualShapeIndex"] = obstacle_visual_id
        if not visual_only:
            obstacle_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=cuboid.half_extents,
                physicsClientId=self.clid,
            )
            kwargs["baseCollisionShapeIndex"] = obstacle_collision_id
        obstacle_id = p.createMultiBody(
            basePosition=cuboid.center,
            baseOrientation=cuboid.pose.so3.xyzw,
            physicsClientId=self.clid,
            **kwargs,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_cylinder(self, cylinder, color=None, visual_only=False):
        assert isinstance(cylinder, Cylinder)
        if color is None:
            color = [0.85882353, 0.14117647, 0.60392157, 1]
        assert not cylinder.is_zero_volume(), "Cannot load zero volume cylinder"
        kwargs = {}
        if self.use_gui:
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=cylinder.radius,
                length=cylinder.height,
                rgbaColor=color,
                physicsClientId=self.clid,
            )
            kwargs["baseVisualShapeIndex"] = obstacle_visual_id
        if not visual_only:
            obstacle_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=cylinder.radius,
                height=cylinder.height,
                physicsClientId=self.clid,
            )
            kwargs["baseCollisionShapeIndex"] = obstacle_collision_id
        obstacle_id = p.createMultiBody(
            basePosition=cylinder.center,
            baseOrientation=cylinder.pose.so3.xyzw,
            physicsClientId=self.clid,
            **kwargs,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_sphere(self, sphere, color=None, visual_only=False):
        assert isinstance(sphere, Sphere)
        if color is None:
            color = [0.0, 0.0, 0.0, 1.0]
        kwargs = {}
        if self.use_gui:
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=sphere.radius,
                rgbaColor=color,
                physicsClientId=self.clid,
            )
            kwargs["baseVisualShapeIndex"] = obstacle_visual_id
        if not visual_only:
            obstacle_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=sphere.radius,
                physicsClientId=self.clid,
            )
            kwargs["baseCollisionShapeIndex"] = obstacle_collision_id
        obstacle_id = p.createMultiBody(
            basePosition=sphere.center,
            physicsClientId=self.clid,
            **kwargs,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_primitives(self, primitives, color=None, visual_only=False):
        ids = []
        for prim in primitives:
            if prim.is_zero_volume():
                continue
            elif isinstance(prim, Cuboid):
                ids.append(self.load_cuboid(prim, color, visual_only))
            elif isinstance(prim, Cylinder):
                ids.append(self.load_cylinder(prim, color, visual_only))
            elif isinstance(prim, Sphere):
                ids.append(self.load_sphere(prim, color, visual_only))
            else:
                raise Exception(
                    "Only cuboids, cylinders, and spheres supported as primitives"
                )
        return ids

    def load_urdf_obstacle(self, path, pose=None):
        if pose is not None:
            obstacle_id = p.loadURDF(
                str(path),
                basePosition=pose.xyz,
                baseOrientation=pose.so3.xyzw,
                useFixedBase=True,
                physicsClientId=self.clid,
            )
        else:
            obstacle_id = p.loadURDF(
                str(path),
                useFixedBase=True,
                physicsClientId=self.clid,
            )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def clear_obstacle(self, id):
        """
        Removes a specific obstacle from the environment

        :param id: Bullet id of obstacle to remove
        """
        if id is not None:
            p.removeBody(id, physicsClientId=self.clid)
            self.obstacle_ids = [x for x in self.obstacle_ids if x != id]

    def clear_all_obstacles(self):
        """
        Removes all obstacles from bullet environment
        """
        for id in self.obstacle_ids:
            if id is not None:
                p.removeBody(id, physicsClientId=self.clid)
        self.obstacle_ids = []


class BulletController(Bullet):
    def __init__(self, gui=False, hz=12, substeps=20):
        """
        :param gui: Whether to use a gui to visualize the environment.
                    Only one gui instance allowed
        """
        super().__init__(gui)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1 / hz,
            numSubSteps=substeps,
            deterministicOverlappingPairs=1,
            physicsClientId=self.clid,
        )

    def step(self):
        p.stepSimulation(physicsClientId=self.clid)


class BulletRobot:
    def __init__(self, clid, **kwargs):
        self.clid = clid
        # TODO implement collision free robot
        self.id = self.load(clid)
        self._setup_robot()
        self._set_robot_specifics(**kwargs)

    def load(self, clid):
        urdf = str(self.robot_type.urdf_bullet_path)
        return p.loadURDF(
            urdf,
            useFixedBase=False, # for meckinova, useFixedBase must be false
            physicsClientId=clid,
            flags=p.URDF_USE_SELF_COLLISION,
        )

    @property
    def links(self):
        """
        :return: The names and bullet ids of all links for the loaded robot
        """
        return [(k, v) for k, v in self._link_name_to_index.items()]

    def link_id(self, name):
        """
        :return: The bullet id corresponding to a specific link name
        """
        return self._link_name_to_index[name]

    def link_name(self, id):
        """
        :return: The name corresponding to a particular bullet id
        """
        return self._index_to_link_name[id]

    @property
    def link_frames(self):
        """
        :return: A dictionary where the link names are the keys
            and the values are the correponding poses as reflected
            by the current state of the environment
        """
        ret = p.getLinkStates(
            self.id,
            list(range(len(self.links) - 1)),
            computeForwardKinematics=True,
            physicsClientId=self.clid,
        )
        frames = {}
        for ii, r in enumerate(ret):
            frames[self.link_name(ii)] = SE3(
                xyz=np.array(r[4]),
                quaternion=Quaternion([r[5][3], r[5][0], r[5][1], r[5][2]]),
            )
        return frames

    def _setup_robot(self):
        """
        Internal function for setting up the correspondence
        between link names and ids.
        """
        # Code snippet borrowed from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12728
        self._link_name_to_index = {
            p.getBodyInfo(self.id, physicsClientId=self.clid)[0].decode("UTF-8"): -1
        }
        for _id in range(p.getNumJoints(self.id, physicsClientId=self.clid)):
            _name = p.getJointInfo(self.id, _id, physicsClientId=self.clid)[12].decode(
                "UTF-8"
            )
            self._link_name_to_index[_name] = _id
        self._index_to_link_name = {}

        for k, v in self._link_name_to_index.items():
            self._index_to_link_name[v] = k

    def _set_robot_specifics(self, **kwargs):
        raise NotImplemented("Must be set in the robot specific class")


class BulletMecKinova(BulletRobot):
    """
    MecKinova Joints ID:
    [('world', -1), ('virtual_base_x', 0), ('virtual_base_y', 1), ('virtual_base_theta', 2), 
    ('virtual_base_center', 3), ('base_link', 4), ('base_link_arm', 5), ('shoulder_link', 6), 
    ('half_arm_1_link', 7), ('half_arm_2_link', 8), ('forearm_link', 9), 
    ('spherical_wrist_1_link', 10), ('spherical_wrist_2_link', 11), ('bracelet_link', 12), 
    ('end_effector_link', 13), ('camera_link', 14), ('camera_depth_frame', 15), 
    ('camera_color_frame', 16), ('robotiq_arg2f_base_link', 17), ('left_outer_knuckle', 18), 
    ('left_outer_finger', 19), ('left_inner_finger', 20), ('left_inner_finger_pad', 21), 
    ('left_inner_knuckle', 22), ('right_outer_knuckle', 23), ('right_outer_finger', 24), 
    ('right_inner_finger', 25), ('right_inner_finger_pad', 26), ('right_inner_knuckle', 27)]
    """

    JOINTS_NAMES = [
        'base_y_base_x', 
        'base_theta_base_y', 
        'base_link_base_theta', 
        'joint_1',
        'joint_2', 
        'joint_3', 
        'joint_4', 
        'joint_5', 
        'joint_6', 
        'joint_7'
    ]

    def __init__(self, clid, **kwargs):
        self.robot_type = MecKinova
        super().__init__(clid, **kwargs)
    
    @property
    def joint_index(self):
        return self.get_joint_index()

    def _set_robot_specifics(self, default_prismatic_value=0.04):
        self.default_prismatic_value = default_prismatic_value

    def marionette(self, state, velocities=None):
        if velocities is None:
            velocities = [0.0 for _ in state]
        assert len(state) == len(velocities)
        for i in range(0, MecKinova.DOF):
            p.resetJointState(
                self.id,
                self.joint_index[i],
                state[i],
                targetVelocity=velocities[i],
                physicsClientId=self.clid,
            )
    
    def get_joint_infos(self):
        """
        :return List[tuple]: [(jointIndex, jointName, ...), ...]
        """
        infos = []
        for i in range(0, p.getNumJoints(self.id)):
            infos.append(p.getJointInfo(self.id, i, physicsClientId=self.clid))
        return infos
    
    def get_joint_index(self):
        """
        Get the iondex of the joints in JOINTS_NAMES.
        NOTE The joints have to be in the same order as JOINTS_NAMES.
        """
        index = []
        joints_infos = self.get_joint_infos()
        for l in joints_infos:
            if l[1].decode('utf-8') in BulletMecKinova.JOINTS_NAMES:
                index.append(l[0])
        return index

    def get_joint_states(self):
        """
        :return: (joint positions, joint velocities)
        """
        states = p.getJointStates(
            self.id, self.joint_index, physicsClientId=self.clid
        )
        return [s[0] for s in states], [s[1] for s in states]

    def control_position(self, state):
        assert len(state) in [10]
        """
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPositions=state,
            targetVelocities=[0] * len(state),
            forces=[250] * len(state),
            positionGains=[0.01] * len(state),
            velocityGains=[1.0] * len(state),
            physicsClientId=self.clid,
        )
        """
        velocities = [0.0 for _ in state]
        for i in range(0, 10):
            p.resetJointState(
                self.id,
                self.joint_index[i],
                state[i],
                targetVelocity=velocities[i],
                physicsClientId=self.clid,
            )
    
    def closest_distance_to_self(self, max_radius):
        contacts = p.getClosestPoints(
            self.id, self.id, max_radius, physicsClientId=self.clid
        )
        # Manually filter out fixed connections that shouldn't be considered
        # TODO fix this somehow
        filtered = []
        for c in contacts:
            # A link is always in collision with itself and its neighbors
            if abs(c[3] - c[4]) <= 1:
                continue
            # The links in Mec and the links in end-effecter are in collision others
            if c[3] < 5 and c[4] < 5:
                continue
            if c[3] > 10 and c[4] > 10:
                continue
            filtered.append(c)
        if len(filtered):
            return min([x[8] for x in filtered])
        return None

    def closest_distance(self, obstacles, max_radius):
        distances = []
        for id in obstacles:
            closest_points = p.getClosestPoints(
                self.id, id, max_radius, physicsClientId=self.clid
            )
            distances.append([x[8] for x in closest_points])
        if len(distances):
            return min(distances)
        return None

    def in_collision(self, obstacles, radius=0.0, check_self=False):
        """
        Checks whether the robot is in collision with the environment

        :return: Boolean
        """
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        # TODO do some more verification on the contact points
        if check_self:
            if radius > 0.0:
                contacts = p.getClosestPoints(
                    self.id, self.id, radius, physicsClientId=self.clid
                )
            else:
                contacts = p.getContactPoints(
                    self.id, self.id, physicsClientId=self.clid
                )
            # Manually filter out fixed connections that shouldn't be considered
            # TODO fix this somehow
            filtered = []
            for c in contacts:
                # Contact distance, positive for separation
                if c[8] > 0:
                    continue
                # A link is always in collision with itself and its neighbors
                if abs(c[3] - c[4]) <= 1:
                    continue
                if c[3] < 5 and c[4] < 5:
                    continue
                if c[3] > 10 and c[4] > 10:
                    continue
                filtered.append(c)
            if len(filtered) > 0:
                return True

        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            if radius > 0.0:
                contacts = p.getClosestPoints(
                    self.id, id, radius, physicsClientId=self.clid
                )
            else:
                contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            for c in contacts:
                # Contact distance, negative for penetration
                if c[8] < 0:
                    return True
        return False

    def get_collision_points(self, obstacles, check_self=False):
        """
        Checks whether the robot is in collision with the environment

        :return: Boolean
        """
        points = []
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        if check_self:
            contacts = p.getContactPoints(self.id, self.id, physicsClientId=self.clid)
            # Manually filter out fixed connections that shouldn't be considered
            # TODO fix this somehow
            filtered = []
            for c in contacts:
                # Contact distance, positive for separation
                if c[8] > 0:
                    continue
                # A link is always in collision with itself and its neighbors
                if abs(c[3] - c[4]) <= 1:
                    continue
                if c[3] < 5 and c[4] < 5:
                    continue
                if c[3] > 10 and c[4] > 10:
                    continue
                filtered.append(c)
            points.extend([p[5] for p in filtered])

        # Iterate through all obstacles to check for collisions
        if isinstance(obstacles, List):
            for id in obstacles:
                contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
                points.extend([p[5] for p in contacts if p[8] < 0])
        elif isinstance(obstacles, int):
            contacts = p.getContactPoints(self.id, obstacles, physicsClientId=self.clid)
            points.extend([p[5] for p in contacts if p[8] < 0])
        return points

    def get_deepest_collision(self, obstacles):
        distances = []
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            distances.extend([p[8] for p in contacts if p[8] < 0])

        if len(distances) > 0:
            # Distance will be negative if it's a true penetration
            deepest_collision = min(distances)
            if deepest_collision < 0:
                return abs(deepest_collision)
        return 0

    def get_collision_depths(self, obstacles):
        distances = []
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            distances.extend([p[8] for p in contacts])
        return [abs(d) for d in distances if d < 0]