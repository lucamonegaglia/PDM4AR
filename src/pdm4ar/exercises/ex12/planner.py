from mimetypes import init
import random
from dataclasses import dataclass
from typing import Sequence, List, Tuple
from venv import create

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
import scipy as sp
from scipy.interpolate import CubicSpline
from itertools import product

from shapely import Polygon

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
        sg: VehicleGeometry,
    ):
        self.lanelet_network = lanelet_network
        self.player_name = player_name
        self.planning_goal = planning_goal
        self.center_lines = self.compute_center_lines_coefficients()
        self.sg = sg

    def update_sim_obs(self, sim_obs: SimObservations):
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
        distances = np.linalg.norm(center_vertices - ego_position, axis=1)
        closest_index = np.argmin(distances)

        # Ensure ego_position is between the center_vertices
        if (
            np.linalg.norm(center_vertices[0] - ego_position) + self.sim_obs.players["Ego"].state.vx
            < lanelet.distance[-1]
        ):
            s_start = np.linalg.norm(center_vertices[0] - ego_position) + self.sim_obs.players["Ego"].state.vx
        else:
            s_start = 0

        if s_start + self.sim_obs.players["Ego"].state.vx * 5 < lanelet.distance[-1]:
            s_end = s_start + self.sim_obs.players["Ego"].state.vx * 5
        else:
            s_end = lanelet.distance[-1]
        # print("Centers", lanelet.center_vertices)

        # Sample evenly spaced points starting from s_start
        s = np.linspace(
            s_start,
            s_end,
            num_points,
        )

        # Sample points until end of the lanelet, solo per visualizzare perdonami genny
        # s = np.linspace(s_start, lanelet.distance[-1], num_points)

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

        # Ensure x_coords are in increasing order
        reverse = False
        if not np.all(np.diff(x_coords) > 0):
            print("x_coords are not in increasing order")
            reverse = True
            x_coords = x_coords[::-1]
            bc_type = ((1, -bc_value_init), (1, -bc_value_end))
        else:
            bc_type = ((1, bc_value_init), (1, bc_value_end))
        # Create the cubic spline
        cubic_spline = CubicSpline(x_coords, y_coords, bc_type=bc_type)

        # Coefficients of the spline for the smoothness term
        # coefficients = cubic_spline.c
        # Generate discretized points along the spline
        xs = np.linspace(min(x_coords), max(x_coords), 20)  # 20 discretized points
        ys = cubic_spline(xs)

        if reverse:
            xs = xs[::-1]
            ys = ys[::-1]

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
        # TODO merge più spline consecutive, qui sono solo due. oppure dynamic programming, ogni spline ha il suo stage cost e si cerca best path
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
        min_objective_value = float("inf")
        best_path = None
        for path in all_paths:
            objective_value = self.objective_function(path.center_vertices)
            if objective_value < min_objective_value:
                min_objective_value = objective_value
                best_path = path
        # TODO, implement the logic to select the best spline
        # return random path
        return best_path

    def objective_function(self, spline: List[Tuple[float, float]]) -> float:
        objective_value = 0.0
        for i in range(len(spline)):
            # Smoothness term
            # TODO I'd like to do this analytically with the coefficients of the spline and evaluating the integrals but
            # an appropriate data structure would be needed
            # Collision term
            if False:
                objective_value += 1000  # TODO Implement collision module
            # Guidance term
            if self.lanelet_network.find_lanelet_by_position([spline[i]]) == [
                []
            ]:  # Why isn't the point in the lanelet?
                continue
            lane_id = self.lanelet_network.find_lanelet_by_position([spline[i]])[0][0]
            a, b, c = self.center_lines[lane_id]
            distance = self.distance_point_to_line_2d(a, b, c, spline[i][0], spline[i][1])
            objective_value += (
                distance  # not super correct, should be an integra (as long as we compute intergrals is okay??)
            )
        return objective_value

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

    def compute_center_lines_coefficients(self):
        """
        Compute a dictionary of lanelet line coefficients (a, b, c) for each lanelet.

        Parameters:
        lanelet_network: An object containing lanelets, where each lanelet has an ID and center points.

        Returns:
        dict: A dictionary where keys are lanelet IDs and values are the coefficients (a, b, c) of the line
            passing through the first and last center points.
        """
        lanelet_lines = {}

        for lanelet in self.lanelet_network.lanelets:
            lanelet_id = lanelet.lanelet_id
            center_points = np.array(lanelet.center_vertices)

            # Extract the first and last points
            p1 = center_points[0]
            p2 = center_points[-1]

            # Compute the line coefficients
            # Line equation: ax + by + c = 0
            # a = y2 - y1
            # b = x1 - x2
            # c = x2*y1 - x1*y2
            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = p2[0] * p1[1] - p1[0] * p2[1]

            # Store the coefficients in the dictionary
            lanelet_lines[lanelet_id] = [a, b, c]
        return lanelet_lines

    def distance_point_to_line_2d(self, a, b, c, x1, y1):
        """
        Compute the distance from a point to a line in 2D general form.

        Parameters:
        a, b, c (float): Coefficients of the line ax + by + c = 0.
        x1, y1 (float): Coordinates of the point.

        Returns:
        float: The perpendicular distance from the point to the line.
        """
        # Compute the distance using the general formula
        distance = abs(a * x1 + b * y1 + c) / np.sqrt(a**2 + b**2)
        return distance

    def predict_other_cars_positions(self, splines) -> List[Polygon]:
        """
        Predict the future positions of static obstacles.

        Parameters:
        sim_obs (SimObservations): The current simulation observations.

        Returns:
        List[StaticObstacle]: A list of static obstacles with predicted positions.
        """
        ego_x = self.sim_obs.players["Ego"].state.x
        ego_y = self.sim_obs.players["Ego"].state.y
        ego_vx = self.sim_obs.players["Ego"].state.vx
        t = [0]
        self.cars = {}
        self.cars["Ego"] = []
        spline = splines.center_vertices
        for i in range(len(spline) - 1):
            distance = np.linalg.norm(spline[i] - spline[i + 1])
            t.append(distance / ego_vx)
            a = spline[i + 1][1] - spline[i][1]
            b = spline[i][0] - spline[i + 1][0]
            m = -a / b
            theta = np.arctan(m)
            self.cars["Ego"].append(self.create_oriented_rectangle(spline[i], theta))
        print("LE MACCHINEEEEEEEE: ", self.cars)
        self.plot_polygons(self.cars)

    def create_oriented_rectangle(self, center, theta):
        """
        Create a Shapely polygon representing a rectangle centered at a given point, with specified width, length, and orientation.

        Args:
            center (tuple): Coordinates of the rectangle center (x, y).
            w_half (float): Half of the rectangle's width.
            lf (float): Distance from the center to the front.
            lr (float): Distance from the center to the rear.
            theta (float): Orientation of the rectangle in radians.

        Returns:
            shapely.geometry.Polygon: Oriented rectangle as a polygon.
        """
        # Rectangle corners in local coordinates (centered at origin)
        local_corners = np.array(
            [
                [self.sg.lf, self.sg.w_half],  # Front right
                [self.sg.lf, -self.sg.w_half],  # Front left
                [-self.sg.lr, -self.sg.w_half],  # Rear left
                [-self.sg.lr, self.sg.w_half],  # Rear right
            ]
        )

        # Rotation matrix for the given orientation
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Rotate and translate the corners to global coordinates
        global_corners = np.dot(local_corners, rotation_matrix.T) + np.array(center)

        # Create the Shapely polygon
        return Polygon(global_corners)

    def plot_polygons(self, cars_dict, key="Ego", title="Shapely Polygons from Dictionary"):
        """
        Plots a list of Shapely polygons stored under a specific key in a dictionary.

        Args:
            cars_dict (dict): Dictionary containing lists of Shapely Polygon objects.
            key (str): The key to access the list of polygons in the dictionary.
            title (str): Title of the plot.
        """
        if key not in cars_dict:
            raise KeyError(f"The key '{key}' is not in the dictionary.")

        polygons = cars_dict[key]

        # Ensure that the list contains Shapely Polygons
        for idx, poly in enumerate(polygons):
            if not isinstance(poly, Polygon):
                raise TypeError(f"Object at index {idx} is not a Shapely Polygon: {type(poly)}")

        # Function to plot a single polygon
        def plot_polygon(ax, polygon, color="blue"):
            x, y = polygon.exterior.xy
            ax.plot(x, y, color=color, linewidth=1.5)  # Plot outline
            ax.fill(x, y, color=color, alpha=0.4)  # Fill interior

        # Create a plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot each polygon with random colors
        for poly in polygons:
            plot_polygon(ax, poly, color=(random.random(), random.random(), random.random()))

        # Set plot limits and labels
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")  # Ensure equal scaling for x and y axes

        # Display the plot
        plt.savefig("LE.png")


# Example usage
# planner = Planner(lanelet_network, player_name, planning_goal, sim_obs)
# lane_id = 1  # Example lane ID
# num_points = 10
# sampled_points = planner.sample_points_on_lane(lane_id, num_points)
# discretized_spline_points = planner.get_discretized_spline(sampled_points)
# planner.plot_sampled_points(sampled_points, lane_id, discretized_spline_points)
