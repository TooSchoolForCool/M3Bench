import numpy as np
import time
import pickle
from pathlib import Path
import itertools
from eval.sparc import sparc
from utils.transform import SE3, SO3
from termcolor import colored
import logging
from geometrout.primitive import Cuboid, Sphere, Cylinder
from typing import Sequence, Union, List, Tuple, Any, Dict, Optional
from env.agent.mec_kinova import MecKinova
from env.sim.bullet_simulator import Bullet
from cprint import *
import pybullet as p


def percent_true(arr: Sequence) -> float:
    """
    Returns the percent true of a boolean sequence or the percent nonzero of a numerical sequence

    :param arr Sequence: The input sequence
    :rtype float: The percent
    """
    return 100 * np.count_nonzero(arr) / len(arr)


class Evaluator:
    """
    This class can be used to evaluate a whole set of environments and data
    """

    def __init__(self, gui: bool = False):
        """
        Initializes the evaluator class
        :param gui bool: Whether to visualize trajectories (and showing visually whether they are failures)
        """
        self.sim = Bullet(gui=False)
        self.sim_robot = self.sim.load_robot(MecKinova)
        self.robot = MecKinova()

        self.self_collision_sim = Bullet(gui=False)
        self.self_collision_robot = self.self_collision_sim.load_robot(MecKinova)
        
        self.groups = {}
        self.current_result = {}
        self.current_group = None
        self.current_group_key = None

        if gui:
            self.gui_sim = Bullet(gui=True)
            p.configureDebugVisualizer(
                flag=p.COV_ENABLE_WIREFRAME, 
                enable=1, 
                physicsClientId=self.gui_sim.clid
            )
            self.gui_robot = self.gui_sim.load_robot(MecKinova)
        else:
            self.gui_sim = None
            self.gui_robot = None

    def run_visuals(
        self,
        dt: float,
        trajectory: Sequence[Union[Sequence, np.ndarray]],
        obstacles_path: str,
        obstacles_pose:Optional[SE3]=None,
    ):
        """
        Visualizes a trajectory and changes the color based on whether its a physical_success or failure

        :param trajectory Sequence[Union[Sequence, np.ndarray]]: The trajectory to visualize
        :param dt float: The approximate timestep to use when visualizing
        :param target SE3: The target pose (in right gripper frame)
        :param obstacles Union[Cuboid, Cylinder, Sphere]: The obstacles to visualize
        :param physical_success bool: Whether the trajectory was a physical_success
        """
        self.gui_sim.clear_all_obstacles()
        collision_ids = self.gui_sim.load_urdf_obstacle(obstacles_path, obstacles_pose)

        for i, q in enumerate(trajectory):
            self.gui_robot.marionette(q)
            collision_points = self.gui_robot.get_collision_points(
                collision_ids, check_self=False
            )
            if len(collision_points):
                collision_points = np.array(collision_points, dtype=np.double)
                self.gui_sim.load_primitives(
                    [Sphere(center=c, radius=0.02) for c in collision_points],
                    [1, 0, 0, 0.5],
                )
                time.sleep(0.25)
            time.sleep(dt)

    def create_new_group(self, key: str):
        """
        Creates a new metric group (for a new setting, for example)

        :param key str: The key for this metric group
        """
        self.groups[key] = {}
        self.current_group_key = key
        self.current_group = self.groups[key]

    def in_collision(self, trajectory: Sequence[Union[Sequence, np.ndarray]]) -> bool:
        """
        Checks whether the trajectory is in collision according to all including
        collision checkers (using AND between different methods)

        :param trajectory Sequence[Union[Sequence, np.ndarray]]: The trajectory
        :rtype bool: Whether there is a collision
        """
        for i, q in enumerate(trajectory):
            self.sim_robot.marionette(q)
            if self.sim.in_collision(
                self.sim_robot, check_self=True
            ):
                return True
        return False

    def has_self_collision(self, trajectory: Sequence[Union[Sequence, np.ndarray]]) -> bool:
        """
        Checks whether there is a self collision

        :param trajectory Sequence[Union[Sequence, np.ndarray]]: The trajectory
        :rtype bool: Whether there is a self collision
        """
        for i, q in enumerate(trajectory):
            self.self_collision_robot.marionette(q)
            if self.self_collision_sim.in_collision(
                self.self_collision_robot, check_self=True
            ):
                return True
        return False

    def get_collision_depths(self, trajectory: Sequence[Union[Sequence, np.ndarray]]) -> List[float]:
        """
        Get all the collision depths for a trajectory (sometimes can report strange
        values due to inconsistent Bullet collision checking)

        :param trajectory Sequence[Union[Sequence, np.ndarray]]: The trajectory
        :rtype List[float]: A list of all collision depths
        """
        all_depths = []
        for i, q in enumerate(trajectory):
            self.sim_robot.marionette(q)
            depths = self.sim_robot.get_collision_depths(self.sim.obstacle_ids)
            all_depths.extend(depths)
        return all_depths

    @staticmethod
    def violates_joint_limits(trajectory: Sequence[Union[Sequence, np.ndarray]]) -> bool:
        """
        Checks whether any configuration in the trajectory violates joint limits

        :param trajectory Sequence[Union[Sequence, np.ndarray]]: The trajectory
        :rtype bool: Whether there is a joint limit violation
        """
        for i, q in enumerate(trajectory):
            if not MecKinova.within_limits(q):
                return True
        return False

    def has_physical_violation(
        self, trajectory: Sequence[Union[Sequence, np.ndarray]], 
    ) -> bool:
        """
        Checks whether there is any physical violation (collision, self collision, joint limit violation)

        :param trajectory Sequence[Union[Sequence, np.ndarray]]: The trajectory
        :param obstacles List[Union[Cuboid, Cylinder, Sphere]: The obstacles in the scene
        :rtype bool: Whether there is at least one physical violation
        """
        return (
            self.in_collision(trajectory[:-1,:])
            or self.violates_joint_limits(trajectory)
            or self.has_self_collision(trajectory)
        )

    @staticmethod
    def calculate_smoothness(
        agent_object: object,
        trajectory: Sequence[Union[Sequence, np.ndarray]], 
        dt: float
    ) -> Tuple[float, float]:
        """
        Calculate trajectory smoothness using SPARC

        :param trajectory Sequence[Union[Sequence, np.ndarray]]: The trajectory
        :param dt float: The timestep in between consecutive steps of the trajectory
        :rtype Tuple[float, float]: The SPARC in configuration space and end effector space
        """
        assert agent_object is MecKinova, "Agent type is invalid"
        mec_kinova = MecKinova()
        configs = np.asarray(trajectory)
        assert configs.ndim == 2 and configs.shape[1] == MecKinova.DOF
        config_movement = np.linalg.norm(np.diff(configs, 1, axis=0) / dt, axis=1)
        assert len(config_movement) == len(configs) - 1
        config_sparc, _, _ = sparc(config_movement, 1.0 / dt)

        eff_positions = np.asarray(
            [mec_kinova.get_eff_pose(q)[:3, 3] for q in trajectory]
        )
        assert eff_positions.ndim == 2 and eff_positions.shape[1] == 3
        eff_movement = np.linalg.norm(np.diff(eff_positions, 1, axis=0) / dt, axis=1)
        assert len(eff_movement) == len(eff_positions) - 1
        eff_sparc, _, _ = sparc(eff_movement, 1.0 / dt)

        return config_sparc, eff_sparc

    def calculate_eff_path_lengths(self, agent_object: object, trajectory: Sequence[Union[Sequence, np.ndarray]]) -> Tuple[float, float]:
        """
        Calculate the end effector path lengths (position and orientation).
        Orientation is in degrees.

        :param trajectory Sequence[Union[Sequence, np.ndarray]]: The trajectory
        :rtype Tuple[float, float]: The path lengths (position, orientation)
        """
        assert agent_object is MecKinova, "Agent type is invalid"
        eff_poses = [SE3(self.robot.get_eff_pose(q)) for q in trajectory]
        eff_positions = np.asarray([pose._xyz for pose in eff_poses])
        assert eff_positions.ndim == 2 and eff_positions.shape[1] == 3
        position_step_lengths = np.linalg.norm(
            np.diff(eff_positions, 1, axis=0), axis=1
        )
        eff_position_path_length = sum(position_step_lengths)

        eff_quaternions = [pose.so3._quat for pose in eff_poses]
        eff_orientation_path_length = 0
        for qi, qj in zip(eff_quaternions[:-1], eff_quaternions[1:]):
            eff_orientation_path_length += np.abs(
                np.degrees((qj * qi.conjugate).radians)
            )
        return eff_position_path_length, eff_orientation_path_length

    def evaluate_trajectory(
        self,
        dt: float,
        time: float,
        trajectory: Sequence[Union[Sequence, np.ndarray]],
        agent_object: object,
        obstacles_path: str,
        obstacles_pose: Optional[SE3]=None,
        skip_metrics: bool = False,
    ):
        """
        Evaluates a single trajectory and stores the metrics in the current group.
        Will visualize and print relevant info if `self.gui` is `True`

        :param trajectory Sequence[Union[Sequence, np.ndarray]]: The trajectory
        :param dt float: The time step for the trajectory
        :param target SE3: The target pose
        :param obstacles Union[Cuboid, Cylinder, Sphere]: The obstacles in the scene
        :param target_volume Union[Cuboid, Cylinder, Sphere]: The target volume for the trajectory
        :param target_negative_volumes Union[Cuboid, Cylinder, Sphere]: Volumes that the target should definitely be outside
        :param time float: The time taken to calculate the trajectory
        :param skip_metrics bool: Whether to skip the path metrics (for example if it's a feasibility planner that failed)
        """
        # assert agent_object is MecKinova, "Agent type is invalid"
        def add_metric(key, value):
            self.current_group[key] = self.current_group.get(key, []) + [value]
            self.current_result[key] = value

        if skip_metrics:
            add_metric("physical_success", False)
            add_metric("time", np.inf)
            add_metric("skips", True)
            return
        self.sim.clear_all_obstacles()
        self.sim.load_urdf_obstacle(path=obstacles_path, pose=obstacles_pose)

        # the last frame of picking and placing motion needs collision
        in_collision = self.in_collision(trajectory[:-1,:])
        if in_collision:
            collision_depths = (
                self.get_collision_depths(trajectory[:-1,:]) if in_collision else []
            )
        else:
            collision_depths = []
        add_metric("collision_depths", collision_depths)
        add_metric("collision", in_collision)
        add_metric("joint_limit_violation", self.violates_joint_limits(trajectory))
        add_metric("self_collision", self.has_self_collision(trajectory))

        physical_violation = self.has_physical_violation(trajectory)
        add_metric("physical_violations", physical_violation)

        config_smoothness, eff_smoothness = self.calculate_smoothness(agent_object, trajectory, dt)
        add_metric("config_smoothness", config_smoothness)
        add_metric("eff_smoothness", eff_smoothness)

        (
            eff_position_path_length,
            eff_orientation_path_length,
        ) = self.calculate_eff_path_lengths(agent_object, trajectory)
        add_metric("eff_position_path_length", eff_position_path_length)
        add_metric("eff_orientation_path_length", eff_orientation_path_length)

        physical_success = not physical_violation

        add_metric("physical_success", physical_success)
        add_metric("time", time)
        add_metric("num_steps", len(trajectory))

        def msg(metric, is_error):
            return colored(metric, "red") if is_error else colored(metric, "green")

        if self.gui_sim is not None and len(collision_depths) > 0:
            print("Config SPARC:", msg(config_smoothness, config_smoothness > -1.6))
            print("End Eff SPARC:", msg(eff_smoothness, eff_smoothness > -1.6))
            print(f"End Eff Position Path Length: {eff_position_path_length}")
            print(f"End Eff Orientation Path Length: {eff_orientation_path_length}")
            print("Physical violation:", msg(physical_violation, physical_violation))
            print(
                "Collisions:",
                msg(
                    self.in_collision(trajectory),
                    self.in_collision(trajectory),
                ),
            )
            if len(collision_depths) > 0:
                print(f"Mean Collision Depths: {100 * np.mean(collision_depths)}")
                print(f"Median Collision Depths: {100 * np.median(collision_depths)}")
            print(
                "Joint Limit Violation:",
                msg(
                    self.violates_joint_limits(trajectory),
                    self.violates_joint_limits(trajectory),
                ),
            )
            print(
                "Self collision:",
                msg(
                    self.has_self_collision(trajectory),
                    self.has_self_collision(trajectory),
                ),
            )
            self.run_visuals(dt, trajectory, obstacles_path, obstacles_pose)
        
        return self.current_result # return current trajectory evaluation result


    @staticmethod
    def metrics(group: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates the metrics for a specific group

        :param group Dict[str, Any]: The group of results
        :rtype Dict[str, float]: The metrics
        """
        physical_success = percent_true(group["physical_success"])

        physical = percent_true(group["physical_violations"])
        config_smoothness = np.mean(group["config_smoothness"])
        eff_smoothness = np.mean(group["eff_smoothness"])
        all_eff_position_path_lengths = np.asarray(group["eff_position_path_length"])
        all_eff_orientation_path_lengths = np.asarray(
            group["eff_orientation_path_length"]
        )
        all_times = np.asarray(group["time"])

        is_smooth = percent_true(
            np.logical_and(
                np.asarray(group["config_smoothness"]) < -1.6,
                np.asarray(group["eff_smoothness"]) < -1.6,
            )
        )

        # Only use filtered physical_successes for eff path length because we only
        # save the metric when it's an unskipped trajectory
        # TODO maybe clean this up so that it's not so if/else-y and either
        # all metrics are saved or only the ones that aren't skipped save
        skips = []
        if "skips" in group:
            # Needed when there are skips
            physical_successes = np.asarray(group["physical_success"])
            unskipped_physical_successes = physical_successes[~np.isinf(all_times)]
            skips = group["skips"]
        else:
            unskipped_physical_successes = group["physical_success"]

        physical_success_position_path_lengths = all_eff_position_path_lengths[
            list(unskipped_physical_successes)
        ]

        physical_success_orientation_path_lengths = all_eff_orientation_path_lengths[
            list(unskipped_physical_successes)
        ]
        eff_position_path_length = (
            np.mean(physical_success_position_path_lengths),
            np.std(physical_success_position_path_lengths),
        )

        eff_orientation_path_length = (
            np.mean(physical_success_orientation_path_lengths),
            np.std(physical_success_orientation_path_lengths),
        )
        physical_success_times = all_times[list(group["physical_success"])]
        time = (
            np.mean(physical_success_times),
            np.std(physical_success_times),
        )

        collision = percent_true(group["collision"])
        joint_limit = percent_true(group["joint_limit_violation"])
        self_collision = percent_true(group["self_collision"])
        depths = np.array(
            list(itertools.chain.from_iterable(group["collision_depths"]))
        )
        all_num_steps = np.asarray(group["num_steps"])
        physical_success_num_steps = all_num_steps[list(unskipped_physical_successes)]
        step_time = (
            np.mean(physical_success_times / physical_success_num_steps),
            np.std(physical_success_times / physical_success_num_steps),
        )
        return {
            "physical_success": physical_success,
            "total": len(group["physical_success"]),
            "skips": len(skips),
            "time": time,
            "step time": step_time,
            "env collision": collision,
            "self collision": self_collision,
            "joint violation": joint_limit,
            "physical violations": physical,
            "average collision depth": 100 * np.mean(depths),
            "median collision depth": 100 * np.median(depths),
            "is smooth": is_smooth,
            "average config sparc": config_smoothness,
            "average eff sparc": eff_smoothness,
            "eff position path length": eff_position_path_length,
            "eff orientation path length": eff_orientation_path_length,
        }

    @staticmethod
    def print_metrics(group: Dict[str, Any]):
        """
        Prints the metrics in an easy to read format

        :param group Dict[str, float]: The group of results
        """
        metrics = Evaluator.metrics(group)
        print(f"Total problems: {metrics['total']}")
        print(f"# Skips (Hard Failures): {metrics['skips']}")
        print(f"% Physical_success: {metrics['physical_success']:4.2f}")
        print(f"% With Environment Collision: {metrics['env collision']:4.2f}")
        print(f"% With Self Collision: {metrics['self collision']:4.2f}")
        print(f"% With Joint Limit Violations: {metrics['joint violation']:4.2f}")
        print(f"Average Collision Depth (cm): {metrics['average collision depth']}")
        print(f"Median Collision Depth (cm): {metrics['median collision depth']}")
        print(f"% With Physical Violations: {metrics['physical violations']:4.2f}")
        print(f"Average Config SPARC: {metrics['average config sparc']:4.2f}")
        print(f"Average End Eff SPARC: {metrics['average eff sparc']:4.2f}")
        print(f"% Smooth: {metrics['is smooth']:4.2f}")
        print(
            "Average End Eff Position Path Length:"
            f" {metrics['eff position path length'][0]:4.2f}"
            f" ± {metrics['eff position path length'][1]:4.2f}"
        )
        print(
            "Average End Eff Orientation Path Length:"
            f" {metrics['eff orientation path length'][0]:4.2f}"
            f" ± {metrics['eff orientation path length'][1]:4.2f}"
        )
        print(f"Average Time: {metrics['time'][0]:4.2f} ± {metrics['time'][1]:4.2f}")
        print(
            "Average Time Per Step (Not Always Valuable):"
            f" {metrics['step time'][0]:4.6f}"
            f" ± {metrics['step time'][1]:4.6f}"
        )

    def save_group(self, directory: str, test_name: str, key: Optional[str] = None):
        """
        Save the results of a single group

        :param directory str: The directory in which to save the results
        :param test_name str: The name of this specific test
        :param key Optional[str]: The group key to use. If not specified will use the current group
        """
        if key is None:
            group = self.current_group
        else:
            group = self.groups[key]
        save_path = Path(directory) / f"{test_name}_{self.current_group_key}.pkl"
        print(f"Saving group metrics to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(group, f)

    def save(self, directory: str, test_name: str):
        """
        Save all the groups

        :param directory str: The directory name in which to save results
        :param test_name str: The test name (used as the file name)
        """
        save_path = Path(directory) / f"{test_name}_metrics.pkl"
        print(f"Metrics will save to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(self.groups, f)

    def print_group_metrics(self, key: Optional[str] = None):
        """
        Prints out the metrics for a specific group

        :param key Optional[str]: The group key (if none specified, will use current group)
        """
        if key is not None:
            self.current_group = self.groups[key]
            self.current_group_key = key
        return self.print_metrics(self.current_group)

    def print_overall_metrics(self):
        """
        Prints the metrics for the aggregated results over all groups
        """
        supergroup = {}
        keys = set()
        for group in self.groups.values():
            for key in group.keys():
                keys.add(key)

        for key in keys:
            metrics = []
            for group in self.groups.values():
                metrics.extend(group.get(key, []))
            supergroup[key] = metrics
        return self.print_metrics(supergroup)
