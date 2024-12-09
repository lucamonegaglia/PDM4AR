from mimetypes import init
import random
from dataclasses import dataclass
from typing import Sequence, List, Tuple

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from itertools import product

matplotlib.use("Agg")


class Planner:
    lanelet_network: LaneletNetwork
    player_name: PlayerName
    planning_goal: PlanningGoal
    sim_obs: SimObservations

    def __init__(
        self,
        lanelet_network: LaneletNetwork,
        player_name: PlayerName,
        planning_goal: PlanningGoal,
        sim_obs: SimObservations,
    ):
        self.lanelet_network = lanelet_network
        self.player_name = player_name
        self.planning_goal = planning_goal
        self.sim_obs = sim_obs

    def sample_points_on_lane(self, lane_id: int, num_points: int) -> Sequence[np.ndarray]:
        lanelet = self.lanelet_network.find_lanelet_by_id(lane_id)
        if not lanelet:
            raise ValueError(f"Lanelet with id {lane_id} not found")

        # Get Ego's current position
        ego_position = np.array([self.sim_obs.players["Ego"].state.x, self.sim_obs.players["Ego"].state.y])

        # Project Ego's position onto the centerline to get the starting arc length
        center_vertices = lanelet.center_vertices

        # Find the closest vertex to ego_position
        distance = np.linalg.norm(center_vertices[0] - ego_position)
        # print("Centers", lanelet.center_vertices)

        # Sample evenly spaced points starting from s_start
        # s = np.linspace(
        #     distance + self.sim_obs.players["Ego"].state.vx,
        #     distance + self.sim_obs.players["Ego"].state.vx * 5,
        #     num_points,
        # )

        # Sample points until end of the lanelet, solo per visualizzare perdonami genny
        s = np.linspace(distance + self.sim_obs.players["Ego"].state.vx, lanelet.distance[-1], num_points)

        sampled_points = []
        for i in range(num_points):
            points = lanelet.interpolate_position(
                s[i]
            )  # The interpolated positions on the center/right/left polyline and the segment id
            if i == 0:
                index_init = points[3]
            if i == num_points - 1:
                index_end = points[3]
            sampled_points.append(points[0:3])

        return sampled_points, index_init, index_end

    def get_discretized_spline(
        self, sampled_points: Sequence[np.ndarray], bc_value_init, bc_value_end
    ) -> List[Tuple[float, float]]:
        """
        Generate a cubic spline for three points and discretize it.

        Parameters:
            sampled_points (Sequence[np.ndarray]): Three points, each a numpy array [x, y].

        Returns:
            List[Tuple[float, float]]: Discretized points along the cubic spline.
        """

        # Extract x and y coordinates
        x_coords = [point[0] for point in sampled_points]
        y_coords = [point[1] for point in sampled_points]

        # Create the cubic spline
        bc_type = ((1, bc_value_init), (1, bc_value_end))
        cubic_spline = CubicSpline(x_coords, y_coords, bc_type=bc_type)

        # Generate discretized points along the spline
        xs = np.linspace(min(x_coords), max(x_coords), 20)  # 20 discretized points
        ys = cubic_spline(xs)

        # Return discretized points as a list of tuples
        return list(zip(xs, ys))

    def get_all_discretized_splines(
        self, sampled_points_list: Sequence[Sequence[np.ndarray]], bc_value_init, bc_value_end
    ) -> List[List[Tuple[float, float]]]:
        """
        Generates discretized splines for all combinations of sampled points.
        """
        all_discretized_splines = []

        # Generate Cartesian product of all combinations of center, left, and right points
        # point_combinations = product(*sampled_points_list)  # Cartesian product
        ego_position = np.array([self.sim_obs.players["Ego"].state.x, self.sim_obs.players["Ego"].state.y])

        point_combinations = []
        for i in range(len(sampled_points_list) - 1):
            layer_combinations = list(product(sampled_points_list[i], sampled_points_list[i + 1]))
            point_combinations += layer_combinations
        for i, item in enumerate(sampled_points_list[0]):
            point_combinations.append((ego_position, item))
        # print("Point combinations: ", point_combinations)
        i = 0
        for combination in point_combinations:
            spline_points = self.get_discretized_spline(combination, bc_value_init, bc_value_end)
            all_discretized_splines.append(spline_points)
            i += 1
        print("Number of splines: ", i)
        return all_discretized_splines

    def plot_all_discretized_splines(
        self, all_discretized_splines: List[List[Tuple[float, float]]], filename: str = "all_discretized_splines.png"
    ):
        plt.figure(figsize=(10, 6))
        for spline_points in all_discretized_splines:
            spline_x, spline_y = zip(*spline_points)
            plt.plot(spline_x, spline_y, label="Discretized Spline")

        plt.title("All Discretized Splines")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def get_player_lane_id(self) -> int:
        return self.lanelet_network.find_lanelet_by_position(
            [np.array([self.sim_obs.players["Ego"].state.x, self.sim_obs.players["Ego"].state.y])]
        )[0][0]

    def get_goal_lane_id(self) -> int:
        # Implementation for getting the goal lane ID
        pass

    def get_all_possible_paths(self, all_discretized_splines: List[List[Tuple[float, float]]]) -> List[Lanelet]:
        # Implementation for getting all possible paths
        all_paths = []

        # Merge splines that have common start/end points
        # TODO merge più spline consecutive, qui sono solo due
        merged_splines = []
        for i, spline1 in enumerate(all_discretized_splines):
            for j, spline2 in enumerate(all_discretized_splines):
                if i != j and np.allclose(spline1[-1], spline2[0]):
                    merged_splines.append(spline1 + spline2[1:])

        # Convert merged splines to Lanelet objects
        for merged_spline in merged_splines:
            lanelet = self.get_path_from_waypoints(merged_spline)
            all_paths.append(lanelet)

        return all_paths

    def get_best_path(self, all_discretized_splines: List[List[Tuple[float, float]]]) -> Lanelet:

        # Merge splines to get all possible paths
        all_paths = self.get_all_possible_paths(all_discretized_splines)

        # TODO, implement the logic to select the best spline
        # return random path
        return all_paths[random.randint(0, len(all_paths) - 1)]

    def get_path_from_waypoints(self, waypoints: Sequence[np.ndarray]) -> Lanelet:
        """
        Generate a path (Lanelet object) from waypoints (coming from one spline), treating them as center of the virtual lanelet.
        """
        assert all(
            isinstance(point, tuple) and len(point) == 2 for point in waypoints
        ), "Waypoints must be a list of tuples like [(x0, y0), (x1, y1), ...]"

        center_points = np.asarray(waypoints)
        radius = 1.5  # Example radius for left and right points

        # Calculate direction vectors
        directions = np.diff(center_points, axis=0)
        norms = np.linalg.norm(directions, axis=1).reshape(-1, 1)
        directions /= norms

        # Calculate perpendicular vectors (+pi/2 rotation)
        perp_vectors = np.column_stack((-directions[:, 1], directions[:, 0]))

        # Calculate left and right points
        left_points = center_points[:-1] + radius * perp_vectors
        right_points = center_points[:-1] - radius * perp_vectors

        # Add the last point
        last_perp_vector = perp_vectors[-1]
        left_points = np.vstack([left_points, center_points[-1] + radius * last_perp_vector])
        right_points = np.vstack([right_points, center_points[-1] - radius * last_perp_vector])

        # Create the Lanelet object
        lanelet_ids = [lanelet.lanelet_id for lanelet in self.lanelet_network.lanelets]
        lanelet_id = max(lanelet_ids, default=0) + 1
        lanelet = Lanelet(
            left_vertices=left_points,
            center_vertices=center_points,
            right_vertices=right_points,
            lanelet_id=lanelet_id,  # Set to a unique ID
        )

        return lanelet


# Example usage
# planner = Planner(lanelet_network, player_name, planning_goal, sim_obs)
# lane_id = 1  # Example lane ID
# num_points = 10
# sampled_points = planner.sample_points_on_lane(lane_id, num_points)
# discretized_spline_points = planner.get_discretized_spline(sampled_points)
# planner.plot_sampled_points(sampled_points, lane_id, discretized_spline_points)
