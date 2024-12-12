import heapq
from mimetypes import init
import random
from dataclasses import dataclass
import time
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
from rtree import index
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
        goal_lanelet_id: int,
    ):
        self.lanelet_network = lanelet_network
        self.player_name = player_name
        self.planning_goal = planning_goal
        self.center_lines = self.compute_center_lines_coefficients()
        self.sg = sg
        self.goal_lanelet_id = goal_lanelet_id
        self.cars = {}
        self.sampling_on_goal_lane = False

    def set_sampling_on_goal_lane(self):
        self.sampling_on_goal_lane = True

    def where_is_goal(self):
        a_g, b_g, c_g = self.center_lines[self.goal_lanelet_id]
        print("Goal line: ", a_g, b_g, c_g)
        a_e, b_e, c_e = self.center_lines[self.current_ego_lanelet_id]
        print("Ego line: ", a_e, b_e, c_e)
        if c_g > c_e:
            print("left")
        elif c_g == c_e:
            print("straight")
        else:
            print("right")

    def update_sim_obs(self, sim_obs: SimObservations):
        self.sim_obs = sim_obs
        self.ego_position = np.array([sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y])
        l = self.lanelet_network.find_lanelet_by_position([self.ego_position])[0]
        if not l:
            print("Outside of lane")
            return VehicleCommands(acc=0, ddelta=0)
        self.current_ego_lanelet_id = l[0]

    def sample_points_on_lane(self, lane_id: int, num_points: int) -> Sequence[np.ndarray]:
        lanelet = self.lanelet_network.find_lanelet_by_id(lane_id)
        if not lanelet:
            raise ValueError(f"Lanelet with id {lane_id} not found")

        # Get Ego's current position
        self.ego_position = np.array([self.sim_obs.players["Ego"].state.x, self.sim_obs.players["Ego"].state.y])

        # Project Ego's position onto the centerline to get the starting arc length
        center_vertices = lanelet.center_vertices

        # Ensure self.ego_position is between the center_vertices
        if np.linalg.norm(center_vertices[0] - self.ego_position) + self.sim_obs.players["Ego"].state.vx > np.max(
            lanelet.distance
        ):
            s_start = (
                np.linalg.norm(center_vertices[0] - self.ego_position)
                + (np.max(lanelet.distance) - np.linalg.norm(center_vertices[0] - self.ego_position) - 1) / 3
            )
            print("S_start: ", s_start)
        else:
            s_start = np.linalg.norm(center_vertices[0] - self.ego_position) + self.sim_obs.players["Ego"].state.vx

        if s_start + self.sim_obs.players["Ego"].state.vx * 5 < np.max(lanelet.distance):
            s_end = s_start + self.sim_obs.players["Ego"].state.vx * 5
        else:
            s_end = np.max(lanelet.distance)
            print("S_end: ", s_end)
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
        dict_points_layer = {}
        for i in range(num_points):
            points = lanelet.interpolate_position(
                s[i]
            )  # The interpolated positions on the center/right/left polyline and the segment id
            # Assuming `points` is a tuple/list containing the three points returned by interpolate_position
            center_point = points[0]  # Center of the lane
            right_point = points[1]  # Rightmost point of the lane
            left_point = points[2]  # Leftmost point of the lane

            # Calculate the middle points
            middle_right_point = ((center_point[0] + right_point[0]) / 2, (center_point[1] + right_point[1]) / 2)

            middle_left_point = ((center_point[0] + left_point[0]) / 2, (center_point[1] + left_point[1]) / 2)

            # The new points: center, middle-right, and middle-left
            new_points = (center_point, middle_right_point, middle_left_point)
            for p in new_points:
                # check that p is an array of lenght 2
                if len(p) == 2:
                    dict_points_layer[(p[0], p[1])] = i + 1  # 1-based index because the first point is the ego position
            if i == 0:
                index_init = points[3]
            if i == num_points - 1:
                index_end = points[3]
            sampled_points.append(new_points)

        return sampled_points, index_init, index_end, dict_points_layer

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
            # print("x_coords are not in increasing order")
            reverse = True
            x_coords = -np.array(x_coords)
            bc_type = ((1, np.pi - bc_value_init), (1, np.pi - bc_value_end))
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
            xs = -xs

        # Return discretized points as a list of tuples
        return list(zip(xs, ys))

    def get_all_discretized_splines(
        self, sampled_points_list: Sequence[Sequence[np.ndarray]], bc_value_init, bc_value_end
    ):
        """
        Generates discretized splines for all combinations of sampled points.
        """
        all_discretized_splines = []
        all_discretized_splines_dict = {}  # same structure but with the point combination as key

        # Generate Cartesian product of all combinations of center, left, and right points
        # point_combinations = product(*sampled_points_list)  # Cartesian product
        self.ego_position = np.array([self.sim_obs.players["Ego"].state.x, self.sim_obs.players["Ego"].state.y])

        point_combinations = []
        for i in range(len(sampled_points_list) - 1):
            layer_combinations = list(product(sampled_points_list[i], sampled_points_list[i + 1]))
            point_combinations += layer_combinations
        for i, item in enumerate(sampled_points_list[0]):
            point_combinations.append((self.ego_position, item))
        # print("Point combinations: ", point_combinations)
        i = 0
        for combination in point_combinations:
            spline_points = self.get_discretized_spline(combination, bc_value_init, bc_value_end)
            all_discretized_splines.append(spline_points)
            all_discretized_splines_dict[
                (combination[0][0], combination[0][1], combination[1][0], combination[1][1])
            ] = spline_points
            i += 1
        # print("Number of splines: ", i)
        return all_discretized_splines, all_discretized_splines_dict

    def graph_search(
        self,
        all_discretized_splines_dict: dict,
        sampled_points_list: Sequence[Sequence[np.ndarray]],
        dict_points_layer: dict,
        start_point: Sequence[np.ndarray],
        goal_point: Sequence[np.ndarray],
        vx,
    ):
        # TODO implement graph search for the best path using UCB
        Q = []
        path_to_goal = []
        cost_path_to_goal = 0
        parent_map = {}
        heapq.heappush(Q, (0, start_point))
        cost_map = {(start_point[0], start_point[1]): 0}
        num_layers = len(sampled_points_list)
        # add to dict points layer the start
        dict_points_layer[(start_point[0], start_point[1])] = 0
        while Q:
            cost, current = heapq.heappop(Q)
            if cost > cost_map[(current[0], current[1])]:
                continue
            if (current == goal_point).all():
                # append to path_to_goal the spline between current and current's parent
                parent = parent_map[(current[0], current[1])]
                path_to_goal.append(all_discretized_splines_dict[(parent[0], parent[1], current[0], current[1])])
                cost_path_to_goal += cost
                while (current != start_point).all():
                    parent = parent_map[(current[0], current[1])]
                    path_to_goal.insert(0, all_discretized_splines_dict[(parent[0], parent[1], current[0], current[1])])
                    cost_path_to_goal += cost_map[(current[0], current[1])]
                    current = parent
                if (path_to_goal[0][0] != start_point).all():
                    path_to_goal.insert(0, all_discretized_splines_dict[(start_point, path_to_goal[0][0])])
                break
            if dict_points_layer[(current[0], current[1])] == num_layers and (current != start_point).all():
                # don't have other neighbors
                continue
            set_of_neighbors = sampled_points_list[dict_points_layer[(current[0], current[1])]]
            for i in range(len(set_of_neighbors)):
                dest_point = sampled_points_list[dict_points_layer[(current[0], current[1])]][i]
                new_cost_to_reach = cost + self.objective_function(
                    all_discretized_splines_dict[(current[0], current[1], dest_point[0], dest_point[1])], vx
                )
                # p = sampled_points_list[dict_points_layer[(current[0], current[1])] + 1][i]
                cost_to_reach_neighbor = cost_map.get(
                    (dest_point[0], dest_point[1]), float("inf")
                )  # if not in the map, return inf
                if new_cost_to_reach < cost_to_reach_neighbor:
                    cost_map[(dest_point[0], dest_point[1])] = new_cost_to_reach
                    parent_map[(dest_point[0], dest_point[1])] = current
                    heapq.heappush(Q, (new_cost_to_reach, dest_point))
        return path_to_goal, cost_path_to_goal

    def merge_adjacent_splines(self, splines):
        l = []
        for s in splines:
            l += s[:-1]
        return l

    def plot_all_discretized_splines(
        self,
        all_discretized_splines: List[List[Tuple[float, float]]],
        best_spline: List[List[Tuple[float, float]]],
        filename: str = "all_discretized_splines.png",
    ):
        plt.figure(figsize=(10, 6))
        for spline_points in all_discretized_splines:
            spline_x, spline_y = zip(*spline_points)
            if spline_points in best_spline:
                print("Ho trovato un pezzo della best spline")
                # plot the best spline with a bigger line
                plt.plot(spline_x, spline_y, linewidth=4)
            else:
                plt.plot(spline_x, spline_y, label="Discretized Spline")

        # if len(highlight_spline) > 0:
        #     spline_x, spline_y = zip(*highlight_spline)
        #     plt.plot(spline_x, spline_y, label="Highlighted Spline", linewidth=2.5, color="red")

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
        for spline1 in all_discretized_splines:
            if np.allclose(spline1[0], self.ego_position):
                for spline2 in all_discretized_splines:
                    if np.allclose(spline1[-1], spline2[0]):
                        spline = spline1 + spline2[1:]
                        merged_splines.append(spline)

        # Convert merged splines to Lanelet objects
        for merged_spline in merged_splines:
            lanelet = self.get_path_from_waypoints(merged_spline)
            all_paths.append(lanelet)

        if not all_paths:
            print("No possible paths found")
        return all_paths

    def get_best_path(self, all_discretized_splines: List[List[Tuple[float, float]]]) -> Lanelet:

        # Merge splines to get all possible paths
        all_paths = self.get_all_possible_paths(all_discretized_splines)
        min_objective_value = float("inf")
        best_path = None
        for path in all_paths:
            objective_value = self.objective_function(path.center_vertices, self.sim_obs.players["Ego"].state.vx)
            vx_slow = self.find_vx()
            objective_value_slow = self.objective_function(path.center_vertices, vx_slow)
            if objective_value < min_objective_value:
                min_objective_value = objective_value
                best_path = path
                vx = self.sim_obs.players["Ego"].state.vx
            if objective_value_slow < min_objective_value:
                min_objective_value = objective_value_slow
                best_path = path
                vx = vx_slow

        # TODO, implement the logic to select the best spline
        # return random path
        if min_objective_value > 10000:
            print("Ho preso il muro fratelli!")

        self.plot_path_and_cars(all_discretized_splines, best_path, vx)
        return best_path, vx

    def plot_path_and_cars(
        self, all_discretized_splines: List[List[Tuple[float, float]]], best_path: Lanelet, vx: float
    ):
        plt.figure(figsize=(10, 6))

        # Plot all discretized splines
        for spline_points in all_discretized_splines:
            spline_x, spline_y = zip(*spline_points)
            plt.plot(spline_x, spline_y, label="Discretized Spline", color="gray", alpha=0.5)

        best_path_x, best_path_y = zip(*best_path.center_vertices)
        plt.plot(best_path_x, best_path_y, label="Best Path", color="blue", linewidth=2)
        # self.predict_ego_positions(best_path.center_vertices, vx)
        self.predict_other_cars_positions(best_path.center_vertices, vx)
        # Plot the predicted positions of the cars
        min_dist = float("inf")
        for key, polygons in self.cars.items():
            for i, poly in enumerate(polygons):
                if (
                    np.linalg.norm(np.array(poly.centroid.coords[0]) - np.array(self.cars["Ego"][i].centroid.coords[0]))
                    < min_dist
                    and key != "Ego"
                ):
                    min_dist = np.linalg.norm(
                        np.array(poly.centroid.coords[0]) - np.array(self.cars["Ego"][i].centroid.coords[0])
                    )
                    closest_timestep = i

        # Plot the polygons for the closest timestep
        for key, polygons in self.cars.items():
            poly = polygons[closest_timestep]
            x, y = poly.exterior.xy
            plt.plot(x, y, label=f"{key} Predicted Position", linewidth=1.5)
            plt.fill(x, y, alpha=0.4)

        plt.title("Best Path, and Predicted Car Positions at closest timestep")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.savefig("all_splines_best_path_and_cars.png")
        plt.close()

    def find_vx(self):
        min_dist = float("inf")
        vx = self.sim_obs.players["Ego"].state.vx
        for keys in self.sim_obs.players:
            if keys != "Ego":
                lane_id = self.lanelet_network.find_lanelet_by_position(
                    [np.array([self.sim_obs.players[keys].state.x, self.sim_obs.players[keys].state.y])]
                )[0]
                if not lane_id:
                    continue
                lane_id = lane_id[0]
                if lane_id == self.current_ego_lanelet_id:
                    r_to_car = (
                        np.array([self.sim_obs.players[keys].state.x, self.sim_obs.players[keys].state.y])
                        - self.ego_position
                    )
                    direction_player_lanelet = (
                        self.lanelet_network.find_lanelet_by_id(self.current_ego_lanelet_id).center_vertices[1]
                        - self.lanelet_network.find_lanelet_by_id(self.current_ego_lanelet_id).center_vertices[0]
                    ) / np.linalg.norm(
                        self.lanelet_network.find_lanelet_by_id(self.current_ego_lanelet_id).center_vertices[1]
                        - self.lanelet_network.find_lanelet_by_id(self.current_ego_lanelet_id).center_vertices[0]
                    )

                    signed_dist_to_car = np.dot(r_to_car, direction_player_lanelet)
                    if signed_dist_to_car < min_dist and signed_dist_to_car > 0:
                        min_dist = signed_dist_to_car
                        vx = self.sim_obs.players[keys].state.vx
        return vx

    def objective_function(self, spline: List[Tuple[float, float]], vx) -> float:
        objective_value = 0.0
        # t_predict = time.time()
        self.predict_other_cars_positions(spline, vx)
        # print("Predict time: ", time.time() - t_predict)
        # t_collision = time.time()
        collisions = self.detect_collisions()
        # print("Collision time: ", time.time() - t_collision)
        if any(collisions[timestep] for timestep in collisions):
            objective_value += 1000000
        for i in range(len(spline)):
            # Smoothness term
            # TODO I'd like to do this analytically with the coefficients of the spline and evaluating the integrals but
            # an appropriate data structure would be needed
            # Guidance term
            if self.sampling_on_goal_lane:
                lane_id = self.goal_lanelet_id
            else:
                lane_id = self.current_ego_lanelet_id
            a, b, c = self.center_lines[lane_id]
            distance = self.distance_point_to_line_2d(a, b, c, spline[i][0], spline[i][1])
            objective_value += (
                distance  # not super correct, should be an integral (as long as we don't compute intergrals is okay??)
            )
        objective_value -= vx
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

    def predict_ego_positions(self, spline, vx) -> List[float]:
        """update cars dictionary and return timesteps list"""
        ego_vx = vx
        timesteps = [0]
        a = spline[1][1] - spline[0][1]
        b = spline[0][0] - spline[1][0]
        m = -a / b
        theta = np.arctan(m)
        self.cars["Ego"] = [self.create_oriented_rectangle(spline[0], theta)]
        if ego_vx < 1e-3:
            return timesteps
        for i in range(1, len(spline) - 1):
            distance = np.linalg.norm(
                np.array([spline[i][0], spline[i][1]]) - np.array([spline[i + 1][0], spline[i + 1][1]])
            )
            timesteps.append(distance / ego_vx)

            a = spline[i + 1][1] - spline[i][1]
            b = spline[i][0] - spline[i + 1][0]
            m = -a / b
            theta = np.arctan(m)
            self.cars["Ego"].append(self.create_oriented_rectangle(spline[i], theta))

        return timesteps

    def predict_other_cars_positions(self, spline, vx) -> List[Polygon]:
        """
        Predict the future positions of static obstacles.

        Parameters:
        sim_obs (SimObservations): The current simulation observations.

        Returns:
        List[StaticObstacle]: A list of static obstacles with predicted positions.
        """
        timesteps = self.predict_ego_positions(spline, vx)
        for car in self.sim_obs.players:
            if car != "Ego":
                self.cars[car] = []
                x = self.sim_obs.players[car].state.x
                y = self.sim_obs.players[car].state.y
                vx = self.sim_obs.players[car].state.vx
                psi = self.sim_obs.players[car].state.psi
                for t in timesteps:
                    x = x + vx * t * np.cos(psi)
                    y = y + vx * t * np.sin(psi)
                    self.cars[car].append(self.create_oriented_rectangle([x, y], psi))
        # print("LE MACCHINEEEEEEEE: ", self.cars)
        # self.plot_polygons(self.cars)

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
                [1.1 * self.sg.lf, 1.1 * self.sg.w_half],  # Front right
                [1.1 * self.sg.lf, -1.1 * self.sg.w_half],  # Front left
                [-1.1 * self.sg.lr, -1.1 * self.sg.w_half],  # Rear left
                [-1.1 * self.sg.lr, 1.1 * self.sg.w_half],  # Rear right
            ]
        )

        # Rotation matrix for the given orientation
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Rotate and translate the corners to global coordinates
        global_corners = rotation_matrix @ local_corners.T + np.array(center).reshape(-1, 1)

        # Create the Shapely polygon
        return Polygon(global_corners.T)

    def plot_polygons(self, title_prefix="Shapely Polygons from Dictionary"):
        """
        Plots lists of Shapely polygons stored under all keys in a dictionary.

        Args:
            cars_dict (dict): Dictionary containing lists of Shapely Polygon objects.
            title_prefix (str): Prefix for the title of each plot.
        """
        for key, polygons in self.cars.items():
            # Ensure that the list contains Shapely Polygons
            for idx, poly in enumerate(polygons):
                if not isinstance(poly, Polygon):
                    raise TypeError(f"Object at index {idx} under key '{key}' is not a Shapely Polygon: {type(poly)}")

            # Function to plot a single polygon
            def plot_polygon(ax, polygon, color="blue"):
                x, y = polygon.exterior.xy
                ax.plot(x, y, color=color, linewidth=1.5)  # Plot outline
                ax.fill(x, y, color=color, alpha=0.4)  # Fill interior

            # Create a plot for each key
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot each polygon with random colors
            for poly in polygons:
                plot_polygon(ax, poly, color=(random.random(), random.random(), random.random()))

            # Set plot limits and labels
            title = f"{title_prefix} - {key}"
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.axis("equal")  # Ensure equal scaling for x and y axes

            # Save the plot for this key
            filename = f"{key}_polygons.png"
            plt.savefig(filename)
            print(f"Saved plot for key '{key}' as {filename}")

            # Close the plot to free memory
            plt.close(fig)

    def bounding_boxes_intersect(self, rect1, rect2):
        return (
            rect1[0] < rect2[2]  # rect1's left edge is to the left of rect2's right edge
            and rect1[2] > rect2[0]  # rect1's right edge is to the right of rect2's left edge
            and rect1[1] < rect2[3]  # rect1's bottom edge is below rect2's top edge
            and rect1[3] > rect2[1]  # rect1's top edge is above rect2's bottom edge
        )

    def detect_collisions(self):
        """
        Detects collisions between the "Ego" car and all other cars at each timestep.

        Args:
            cars: Dictionary of cars where keys are car names and values are lists of
                bounding boxes (min_x, min_y, max_x, max_y) for each timestep.

        Returns:
            collisions: Dictionary where each key is a timestep, and the value is a list
                        of car names that collided with the "Ego" car at that timestep.
        """
        # Initialize collision storage
        collisions = {i: [] for i in range(len(self.cars["Ego"]))}

        # Iterate over timesteps
        for timestep in range(len(self.cars["Ego"])):
            ego_rect = self.cars["Ego"][timestep]

            # Build R-tree index for other cars at this timestep
            idx = index.Index()
            car_to_id = {}  # Map from car name to unique ID for reverse lookup
            id_counter = 0

            for car, rects in self.cars.items():
                if car == "Ego":
                    continue
                other_rect = rects[timestep]
                bounds = other_rect.bounds  # Get the bounds of the rectangle
                idx.insert(id_counter, bounds)  # Insert rectangle bounds into the R-tree
                car_to_id[id_counter] = car  # Map the ID to the car name
                id_counter += 1

            # Query potential collisions using R-tree
            potential_collisions = list(idx.intersection(ego_rect.bounds))

            # Verify potential collisions with bounding box intersection
            for car_id in potential_collisions:
                other_rect = self.cars[car_to_id[car_id]][timestep]
                if ego_rect.intersects(other_rect):
                    collisions[timestep].append(car_to_id[car_id])

        return collisions


# Example usage
# planner = Planner(lanelet_network, player_name, planning_goal, sim_obs)
# lane_id = 1  # Example lane ID
# num_points = 10
# sampled_points = planner.sample_points_on_lane(lane_id, num_points)
# discretized_spline_points = planner.get_discretized_spline(sampled_points)
# planner.plot_sampled_points(sampled_points, lane_id, discretized_spline_points)
