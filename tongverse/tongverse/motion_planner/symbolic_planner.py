# pylint: disable=R0912
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from PIL import Image

from tongverse.utils.a_star import AStarPlanner
from tongverse.utils.constant import RootPath

IMAGESIZE = 256
SCALE = 30
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


def xy_to_cor(x_image, y_image, scale=SCALE, image_size=IMAGESIZE):
    x = (x_image - image_size / 2) * scale / (image_size / 2)
    y = (y_image - image_size / 2) * scale / (image_size / 2)
    return x, y


def cor_to_xy(point, scale=SCALE, image_size=IMAGESIZE):
    x, y = point
    x_image = int(int(x / scale * image_size / 2) + image_size / 2)
    y_image = int(int(y / scale * image_size / 2) + image_size / 2)
    return x_image, y_image


# pylint:disable=C0103
def get_final_xy(x, y, top_offset, left_offset):
    return x - top_offset, y - left_offset


def get_reverse_xy(x, y, top_offset, left_offset):
    return x + top_offset, y + left_offset


def convert_to_255(arr):
    return np.where(arr != 0, 255, arr)


def find_zeros_indices(array):
    ox, oy = [], []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j] == 0:
                ox.append(i)
                oy.append(j)
    return ox, oy


def find_nearest_non_zero_points(array, start_point):
    x, y = start_point
    max_distance = 10

    points = []

    for i in range(x - 1, max(x - max_distance - 1, -1), -1):
        if i < 0:
            break
        if array[i, y] != 0:
            points.append((i, y))
            break

    for i in range(x + 1, min(x + max_distance + 1, array.shape[0])):
        if i >= array.shape[0]:
            break
        if array[i, y] != 0:
            points.append((i, y))
            break

    for j in range(y - 1, max(y - max_distance - 1, -1), -1):
        if j < 0:
            break
        if array[x, j] != 0:
            points.append((x, j))
            break

    for j in range(y + 1, min(y + max_distance + 1, array.shape[1])):
        if j >= array.shape[1]:
            break
        if array[x, j] != 0:
            points.append((x, j))
            break

    return points


# cut zeros, return left_offset and top_offset
def find_min_rectangle(array):
    top, bottom, left, right = 0, array.shape[0] - 1, 0, array.shape[1] - 1

    while top < bottom and np.all(array[top, :] == 0):
        top += 1
    while top < bottom and np.all(array[bottom, :] == 0):
        bottom -= 1
    while left < right and np.all(array[:, left] == 0):
        left += 1
    while left < right and np.all(array[:, right] == 0):
        right -= 1

    left_offset = max(left - 1, 0)
    top_offset = max(top - 1, 0)
    return array[top - 1:bottom + 2, left - 1:right + 2], left_offset, top_offset


class SymbolicPlanner:
    """
    Generate and control agents' action for each step, all from state changes.
    Agent can be a block or other object that can change its state.
    """
    action_space = [
        'step_forward',
        'step_back',
        'turn_left',
        'turn_right',
        'pick',
        'place',
        'goto'
    ]

    def __init__(self, env, agent, angle=0, step_distance=1.0,
                 pick_distance=0.5):
        """
        Initializes the SymbolicController.

        Parameters:
            env (Env): The environment instance.
            agent (omni.isaac.core.objects): object instance.
            step_distance (float): One step distance, the measurement standard is
            the distance standard of the Omniverse used.
            pick_distance (float): The farthest distance that an agent can pick
            up an object,the measurement standard is the distance standard of the
            Omniverse used.
        """
        self.agent = agent
        self.env = env
        self.step_distance = step_distance
        self.object_name = None
        self.action = None
        self.angle = angle
        self.pick_distance = pick_distance
        self.object_held = ""

        matrix_modify = np.array([0., 0., angle])
        agent_orient = self.agent.get_world_pose()[1]
        euler_angles = quat_to_euler_angles(agent_orient, degrees=True)
        euler_angles += matrix_modify
        self.agent.set_world_pose(orientation=np.array(
            euler_angles_to_quat(euler_angles, degrees=True)))

    def generate_action(self, dic: Dict[str, object]) \
            -> Tuple[bool, Tuple:None]:  # noqa: PLR0912
        """
        Based on the input parameters, generate corresponding target pose.

        Parameters:
            dic: Dict[str, object]): including action_type(Type of actions,
            such as goto, pick), maybe object_name(action related objects),
            maybe rotate_angle(the angle of turning left or right)...

        Returns:
            Tuple[bool, Tuple]: bool type value represents
            whether it is successful,and the Tuple is the target pose,
            the former is position, while the latter is orientation.
        """
        target_pose = None
        action_type, object_name = "", ""
        if "action_type" in dic:
            action_type = dic["action_type"]
        if "object_name" in dic:
            object_name = dic["object_name"]
        if action_type == "step_forward":
            _, target_pose = self.step_move("step_forward")
        elif action_type == "step_back":
            _, target_pose = self.step_move("step_back")
        elif action_type == "turn_left":
            if "rotate_angle" in dic:
                _, target_pose = self.turn("turn_left", dic["rotate_angle"])
            else:
                _, target_pose = self.turn("turn_left")
        elif action_type == "turn_right":
            if "rotate_angle" in dic:
                _, target_pose = self.turn("turn_right", dic["rotate_angle"])
            else:
                _, target_pose = self.turn("turn_right")
        elif action_type == "turn_around":
            _, target_pose = self.turn("turn around")
        elif action_type == "pick" and object_name:
            _, target_pose = self.pick(object_name)
        elif action_type == "place" and object_name:
            _, target_pose = self.place(object_name)
        elif action_type == "goto" and object_name:
            plan_result = self.plan(object_name)
            if plan_result[0]:
                _, target_pose = self.goto(plan_result)
            else:
                return False, target_pose
        return target_pose

    def turn(self, action: str, rotate_angle: float = 90.) \
            -> Tuple[bool, Tuple]:
        """
        Actions include left and right turns, as well as turning back.

        Parameters:
            action(str): action name.
            rotate_angle(float): The angle of turning left or right.


        Returns:
            Tuple[bool, Tuple]: bool type value represents
            whether it is successful,and the Tuple is the target pose,
            the former is position, while the latter is orientation.
        """
        matrix_modify = []
        rotate_angle = abs(rotate_angle) % 360
        if action == "turn_left":
            matrix_modify = np.array([0., 0., rotate_angle])
            self.angle -= rotate_angle
        elif action == "turn_right":
            matrix_modify = np.array([0., 0., -rotate_angle])
            self.angle += rotate_angle
        elif action == "turn around":
            matrix_modify = np.array([0., 0., -180.])
            self.angle += 180.
        agent_orient = self.agent.get_world_pose()[1]
        euler_angles = quat_to_euler_angles(agent_orient, degrees=True)
        euler_angles += matrix_modify
        self.agent.set_world_pose(orientation=np.array(
            euler_angles_to_quat(euler_angles, degrees=True)))
        target_pose = (None, np.array(
            euler_angles_to_quat(euler_angles, degrees=True)))
        if self.object_held:
            agent_trans = self.agent.get_world_pose()[0]
            agent_x, agent_y = agent_trans[0], agent_trans[1]
            dy = self.pick_distance * math.cos(math.radians(self.angle))
            dx = self.pick_distance * math.sin(math.radians(self.angle))
            agent_x -= dy
            agent_y += dx
            self.env.get_scene().get_object_by_name(self.object_held).set_world_pose(
                [float(agent_x), float(agent_y), 0.5])
        return True, target_pose

    def step_move(self, action: str) -> Tuple[bool, Tuple]:
        """
        Actions include moving forward and back.

        Parameters:
            action(str): action name.

        Returns:
            Tuple[bool, Tuple]: bool type value represents
            whether it is successful,and the Tuple is the target pose,
            the former is position, while the latter is orientation.
        """
        tmp_angle = self.angle
        agent_trans = self.agent.get_world_pose()[0]
        agent_x, agent_y = 0., 0.
        # go forward
        if action == "step_forward":
            tmp_angle = self.angle
            agent_x, agent_y = agent_trans[0], agent_trans[1]
            dy = self.step_distance * math.cos(math.radians(tmp_angle))
            dx = self.step_distance * math.sin(math.radians(tmp_angle))
            agent_x -= dy
            agent_y += dx
            # if self.object_held:
            #     obj_trans = self.env.get_scene().get_object_by_name(
            #         self.object_held).get_world_pose()[0]
            #     obj_x, obj_y = obj_trans[0], obj_trans[1]
            #     obj_x -= dy
            #     obj_y += dx
            #     self.env.get_scene().get_object_by_name(self.object_held). \
            #         set_world_pose([float(obj_x), float(obj_y), 0.5])
        # go back
        elif action == "step_back":
            tmp_angle = self.angle
            agent_x, agent_y = agent_trans[0], agent_trans[1]
            dy = self.step_distance * math.cos(math.radians(tmp_angle))
            dx = self.step_distance * math.sin(math.radians(tmp_angle))
            agent_x += dy
            agent_y -= dx
            # if self.object_held:
            #     obj_trans = self.env.get_scene().get_object_by_name(
            #         self.object_held).get_world_pose()[0]
            #     obj_x, obj_y = obj_trans[0], obj_trans[1]
            #     obj_x += dy
            #     obj_y -= dx
            #     self.env.get_scene().get_object_by_name(self.object_held). \
            #         set_world_pose([float(obj_x), float(obj_y), 0.5])

        # self.agent.set_world_pose([float(agent_x), float(agent_y), agent_trans[2]])
        target_pose = ([float(agent_x), float(agent_y), agent_trans[2]], None)
        return True, target_pose

    def pick(self, object_name: str) -> Tuple[bool, Tuple]:
        """
        Pick objects.

        Parameters:
            object_name(str): The name of the object you want to pick up.

        Returns:
            Tuple[bool, Tuple]: bool type value represents
            whether it is successful,and the Tuple is the target pose,
            the former is position, while the latter is orientation.
        """
        pick_distance_threshold = 600.
        # already holding an object
        if self.object_held:
            return False, (None, None)
        agent_trans = self.agent.get_world_pose()[0]
        obj_prim = self.env.get_scene().get_object_by_name(object_name)
        obj_x, obj_y = obj_prim.get_world_pose()[0][0], obj_prim.get_world_pose()[0][1]
        agent_x, agent_y = agent_trans[0], agent_trans[1]
        # distance between object and agent is too far
        if (agent_x - obj_x) ** 2 + (agent_y - obj_y) ** 2 > pick_distance_threshold:
            return False, (None, None)
        dy = self.pick_distance * math.cos(math.radians(self.angle))
        dx = self.pick_distance * math.sin(math.radians(self.angle))
        agent_x -= dy
        agent_y += dx
        # disable physical simulation
        self.object_held = object_name
        obj_prim.disable_gravity(True)
        obj_prim.set_world_pose([float(agent_x), float(agent_y), 0.5])
        return True, (None, None)

    def place(self, target_name: str) -> Tuple[bool, Tuple]:
        """
        Place the picked up object onto the target object.

        Parameters:
            target_name(str): Target object to be placed.

       Returns:
            Tuple[bool, Tuple]: bool type value represents
            whether it is successful,and the Tuple is the target pose,
            the former is position, while the latter is orientation.
        """
        target_prim = self.env.get_scene().get_object_by_name(target_name)
        target_x, target_y, target_z = target_prim.get_world_pose()[0]
        self.env.get_scene().get_object_by_name(self.object_held).set_world_pose(
            [float(target_x), float(target_y), float(target_z + 0.25)])
        self.env.get_scene().get_object_by_name(self.object_held).disable_gravity(False)
        self.object_held = ""
        return True, (None, None)

    def plan(self, target_name: str):
        """
        Through the a-star algorithm, calculate whether there is a path
         to the target object.

        Parameters:
            target_name(str): Target object to approach.

        Returns:
            tuple: success, target x,y, direction that  agents need to face.
        """
        file_name = str(RootPath.SCENE / self.env.get_scene().
                        name / "walkmap.png")
        image = Image.open(file_name)
        image_array = np.array(image)
        image_array = convert_to_255(image_array)
        image_array, left_offset, top_offset = find_min_rectangle(image_array)
        start = self.agent.get_world_pose()[0][:2]
        x, y = cor_to_xy(start)
        start_xy = get_final_xy(x, y, top_offset, left_offset)
        end = self.env.get_scene().get_object_by_name(
            target_name).get_world_pose()[0][:2]
        x, y = cor_to_xy(end)
        end_xy = get_final_xy(x, y, top_offset, left_offset)
        cand_list = find_nearest_non_zero_points(image_array, end_xy)
        ox, oy = find_zeros_indices(image_array)
        success = False
        gx, gy = 0, 0
        idx = -1
        for cand in cand_list:
            sx, sy = start_xy
            gx, gy = cand  # [m]
            grid_size = 1.0  # [m]
            robot_radius = 0.0
            a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
            _, _, success = a_star.planning(sx, sy, gx, gy)
            idx += 1
            if success:
                break
        if success:
            gx, gy = get_reverse_xy(gx, gy, top_offset, left_offset)
            x, y = xy_to_cor(gx, gy)
            return success, (x, y), idx
        return False, (0, 0), 0

    def goto(self, plan_result) -> Tuple[bool, Tuple]:
        """
        Based on the calculated target position, move the agent.

        Parameters:
            plan_result(tuple): success, target x,y, direction
            that agents need to face..

        Returns:
            Tuple[bool, Tuple]: bool type value represents
            whether it is successful,and the Tuple is the target pose,
            the former is position, while the latter is orientation.
        """
        direction = plan_result[2]
        target_pose = None
        # up
        if direction == UP:
            self.angle = 180
            agent_orient = quat_to_euler_angles(
                np.array(euler_angles_to_quat([0, 0, 180], degrees=True)), degrees=True)
            # self.agent.set_world_pose(
            #     [plan_result[1][0], plan_result[1][1], 1.],
            #     np.array(euler_angles_to_quat(agent_orient, degrees=True)))
            target_pose = ([plan_result[1][0], plan_result[1][1], 1.],
                           np.array(euler_angles_to_quat(agent_orient, degrees=True)))
        # down
        elif direction == DOWN:
            self.angle = 0
            self.agent.set_world_pose([plan_result[1][0], plan_result[1][1], 1.])
            target_pose = ([plan_result[1][0], plan_result[1][1], 1.])
        # left
        elif direction == LEFT:
            self.angle = -90
            agent_orient = quat_to_euler_angles(
                np.array(euler_angles_to_quat([0, 0, -90], degrees=True)), degrees=True)
            # self.agent.set_world_pose(
            #     [plan_result[1][0], plan_result[1][1], 1.],
            #     np.array(euler_angles_to_quat(agent_orient, degrees=True)))
            target_pose = ([plan_result[1][0], plan_result[1][1], 1.],
                           np.array(euler_angles_to_quat(agent_orient, degrees=True)))
        # right
        elif direction == RIGHT:
            self.angle = 90
            agent_orient = quat_to_euler_angles(
                np.array(euler_angles_to_quat([0, 0, 90], degrees=True)), degrees=True)
            # self.agent.set_world_pose(
            #     [plan_result[1][0], plan_result[1][1], 1.],
            #     np.array(euler_angles_to_quat(agent_orient, degrees=True)))
            target_pose = ([plan_result[1][0], plan_result[1][1], 1.],
                           np.array(euler_angles_to_quat(agent_orient, degrees=True)))
        if self.object_held:
            agent_trans = self.agent.get_world_pose()[0]
            obj_prim = self.env.get_scene().get_object_by_name(self.object_held)
            agent_x, agent_y = agent_trans[0], agent_trans[1]
            # distance between object and agent is too far
            dy = self.pick_distance * math.cos(math.radians(self.angle))
            dx = self.pick_distance * math.sin(math.radians(self.angle))
            agent_x -= dy
            agent_y += dx
            obj_prim.set_world_pose([float(agent_x), float(agent_y), 0.5])
        return True, target_pose
