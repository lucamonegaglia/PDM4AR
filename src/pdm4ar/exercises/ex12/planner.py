from mimetypes import init
import random
from dataclasses import dataclass
from typing import Sequence, List, Tuple

from commonroad.scenario.lanelet import LaneletNetwork
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

matplotlib.use("Agg")


@dataclass
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

        sampled_points = []
        s = np.linspace(0, self.sim_obs.players["Ego"].state.vx * 8, num_points)
        for i in range(num_points):
            points = lanelet.interpolate_position(
                s[i]
            )  # The interpolated positions on the center/right/left polyline and the segment id of the polyline where the interpolation takes place in the form ([x_c,y_c],[x_r,y_r],[x_l,y_l], segment_id)
            sampled_points.append(points)

        return sampled_points

    def get_discretized_spline(self, sampled_points: Sequence[np.ndarray]) -> List[Tuple[float, float]]:
        # Extract x and y coordinates for spline creation
        x_coords = [point[0][0] for point in sampled_points]  # Assuming point[0] is the centerline point
        y_coords = [point[0][1] for point in sampled_points]

        # Create a cubic spline
        cs = CubicSpline(x_coords, y_coords)

        # Generate points for the spline
        xs = np.linspace(min(x_coords), max(x_coords), 100)
        ys = cs(xs)

        # Discretized spline points
        discretized_points = list(zip(xs, ys))

        return discretized_points

    def sample_points_on_player_lane(self, num_points: int) -> Sequence[np.ndarray]:
        player_lane_id = self.get_player_lane_id()
        return self.sample_points_on_lane(player_lane_id, num_points)

    def sample_points_on_goal_lane(self, num_points: int) -> Sequence[np.ndarray]:
        goal_lane_id = self.get_goal_lane_id()
        return self.sample_points_on_lane(goal_lane_id, num_points)

    def plot_sampled_points(
        self, sampled_points: Sequence[np.ndarray], lane_id: int, spline_points: List[Tuple[float, float]] = None
    ):
        # Extract x and y coordinates for plotting
        x_coords = [point[0][0] for point in sampled_points]  # Assuming point[0] is the centerline point
        y_coords = [point[0][1] for point in sampled_points]
        x_coords += [point[1][0] for point in sampled_points]
        y_coords += [point[1][1] for point in sampled_points]
        x_coords += [point[2][0] for point in sampled_points]
        y_coords += [point[2][1] for point in sampled_points]

        # Plot the sampled points
        plt.figure(figsize=(10, 6))
        plt.scatter(x_coords, y_coords, c="blue", marker="o", label="Sampled Points")

        # Plot the spline points if provided
        if spline_points:
            spline_x, spline_y = zip(*spline_points)
            plt.plot(spline_x, spline_y, c="red", label="Cubic Spline")

        plt.title(f"Sampled Points on Lane {lane_id}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.savefig(str(lane_id))
        plt.close()

    def get_player_lane_id(self) -> int:
        return self.lanelet_network.find_lanelet_by_position(
            [np.array([self.sim_obs.players["Ego"].state.x, self.sim_obs.players["Ego"].state.y])]
        )[0][0]

    def get_goal_lane_id(self) -> int:
        # Implementation for getting the goal lane ID
        pass


# Example usage
# planner = Planner(lanelet_network, player_name, planning_goal, sim_obs)
# lane_id = 1  # Example lane ID
# num_points = 10
# sampled_points = planner.sample_points_on_lane(lane_id, num_points)
# discretized_spline_points = planner.get_discretized_spline(sampled_points)
# planner.plot_sampled_points(sampled_points, lane_id, discretized_spline_points)
