from calendar import c
import heapq
from hmac import new
from math import isclose
from mimetypes import init
import random
from dataclasses import dataclass
from threading import local
import time
from turtle import left
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
from matplotlib.pylab import sample
from networkx import center
import numpy as np
import scipy as sp
from scipy.interpolate import CubicSpline
from itertools import product
from rtree import index
from shapely import Polygon, equals_exact

matplotlib.use("Agg")
# questo è il codice


class Planner:
    lanelet_network: LaneletNetwork
    player_name: PlayerName
    planning_goal: PlanningGoal
    sim_obs: SimObservations

    def __init__(
        self,
    ):
        pass

    def initialize(
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
        self.sg = sg
        self.goal_lanelet_id = goal_lanelet_id
        self.cars = {}
        self.sampling_on_goal_lane = False
        self.direction = (
            self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).center_vertices[-1]
            - self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).center_vertices[0]
        ) / np.linalg.norm(
            self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).center_vertices[-1]
            - self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).center_vertices[0]
        )
        center_vertice = self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).center_vertices[0]
        left_vertice = self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).left_vertices[0]
        self.radius = np.linalg.norm(center_vertice - left_vertice)
        self.traffic = False
        self.center_lines = {}

    def set_sampling_on_goal_lane(self):
        self.sampling_on_goal_lane = True

    def where_is_goal(self):
        a_g, b_g, c_g = self.center_lines[self.goal_lanelet_id]
        # print("Goal line: ", a_g, b_g, c_g)
        a_e, b_e, c_e = self.center_lines[self.current_ego_lanelet_id]
        # print("Ego line: ", a_e, b_e, c_e)
        # if c_g > c_e:
        #     #print("left")
        # elif c_g == c_e:
        #     print("straight")
        # else:
        #     print("right")

    def update_sim_obs(self, sim_obs: SimObservations):
        self.sim_obs = sim_obs
        self.ego_position = np.array([sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y])
        l = self.lanelet_network.find_lanelet_by_position([self.ego_position])[0]
        if not l:
            print("Outside of lane")
            return VehicleCommands(acc=0, ddelta=0)
        self.current_ego_lanelet_id = l[0]

    def sample_points_on_lane_good(self, lane_id, num_points):
        lanelet = self.lanelet_network.find_lanelet_by_id(lane_id)
        if not lanelet:
            raise ValueError(f"Lanelet with id {lane_id} not found")

        # Get Ego's current position
        self.ego_position = np.array([self.sim_obs.players["Ego"].state.x, self.sim_obs.players["Ego"].state.y])

        a, b, c = self.center_lines[lane_id]
        distance_to_line = self.distance_point_to_line_2d(a, b, c, self.ego_position[0], self.ego_position[1])
        projected_ego_x = self.ego_position[0] - a * distance_to_line / np.sqrt(a**2 + b**2)
        theta = np.arctan2(self.direction[1], self.direction[0])
        x_start = projected_ego_x + max(self.sim_obs.players["Ego"].state.vx, 3 * (self.sg.lr + self.sg.lf)) * np.cos(
            theta
        )
        x_end = x_start + max(self.sim_obs.players["Ego"].state.vx, 3 * (self.sg.lr + self.sg.lf)) * 5 * np.cos(theta)
        x_lin = np.linspace(
            x_start,
            x_end,
            num_points,
        )
        sampled_points = []
        dict_points_layer = {}
        perp_direction = np.array([-self.direction[1], self.direction[0]])
        r = self.radius
        for i, x in enumerate(x_lin):
            center_y = -a / b * x - c / b
            center_point = (x, center_y)
            if lane_id != self.goal_lanelet_id:
                if self.goal_lanelet_id == self.lanelet_network.find_lanelet_by_id(lane_id).adj_right:
                    point_adj = [x - r * perp_direction[0], center_y - r * perp_direction[1]]
                else:
                    point_adj = [x + r * perp_direction[0], center_y + r * perp_direction[1]]
                middle_point = ((center_point[0] + point_adj[0]) / 2, (center_point[1] + point_adj[1]) / 2)
                new_points = [center_point, middle_point]
                for p in new_points:
                    # check that p is an array of lenght 2
                    if len(p) == 2:
                        dict_points_layer[(p[0], p[1])] = (
                            i + 1
                        )  # 1-based index because the first point is the ego position
                sampled_points.append(new_points)
            else:
                new_points = center_point
                dict_points_layer[(center_point[0], center_point[1])] = i + 1
                sampled_points.append([new_points])

        return sampled_points, dict_points_layer

    def get_discretized_spline(self, point_combination: Sequence[np.ndarray], R) -> List[Tuple[float, float]]:
        """
        Generate a cubic spline for three points and discretize it.

        Parameters:
            sampled_points (Sequence[np.ndarray]): Three points, each a numpy array [x, y].

        Returns:
            List[Tuple[float, float]]: Discretized points along the cubic spline.
        """
        # Extract x and y coordinates
        x_coords = []
        y_coords = []
        for p in point_combination:
            rotated_sampled_points = R.T @ np.array(p)
            x_coords.append(rotated_sampled_points[0])
            y_coords.append(rotated_sampled_points[1])

        # Ensure x_coords are in increasing order
        # reverse = False
        if not np.all(np.diff(x_coords) > 0):
            print("x_coords are not in increasing order")
            return []
        #     reverse = True
        #     x_coords = -np.array(x_coords)
        #     bc_type = ((1, np.pi - bc_value_init), (1, np.pi - bc_value_end))
        # else:
        #     bc_type = ((1, bc_value_init), (1, bc_value_end))

        # Create the cubic spline
        # cubic_spline = CubicSpline(x_coords, y_coords, bc_type=bc_type)
        cubic_spline = CubicSpline(x_coords, y_coords, bc_type=((1, 0), (1, 0)))

        # Coefficients of the spline for the smoothness term
        # coefficients = cubic_spline.c
        # Generate discretized points along the spline
        xs = np.linspace(min(x_coords), max(x_coords), 20)  # 20 discretized points
        ys = cubic_spline(xs)

        # if reverse:
        #     xs = -xs
        xs, ys = R @ np.array([xs, ys])
        # Return discretized points as a list of tuples
        return list(zip(xs, ys))

    def get_all_discretized_splines(self, sampled_points_list: Sequence[Sequence[np.ndarray]]):
        """
        Generates discretized splines for all combinations of sampled points.
        """
        all_discretized_splines = []
        all_discretized_splines_dict = {}  # same structure but with the point combination as key
        theta = np.arctan2(self.direction[1], self.direction[0])
        # Generate Cartesian product of all combinations of center, left, and right points
        # point_combinations = product(*sampled_points_list)  # Cartesian product
        self.ego_position = np.array([self.sim_obs.players["Ego"].state.x, self.sim_obs.players["Ego"].state.y])

        point_combinations = []
        for i, item in enumerate(sampled_points_list[0]):
            point_combinations.append((self.ego_position, item))
        for i in range(len(sampled_points_list) - 1):
            layer_combinations = list(product(sampled_points_list[i], sampled_points_list[i + 1]))
            point_combinations += layer_combinations
        # print("Point combinations: ", point_combinations)
        i = 0
        print("Number of splines: ", len(point_combinations))
        for combination in point_combinations:
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rotated_sampled_point_1 = R.T @ combination[0]
            rotated_sampled_point_2 = R.T @ combination[1]
            spline_points = self.get_discretized_spline(combination, R)
            if not spline_points:
                continue
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
        lane_id: int,
    ):
        # TODO implement graph search for the best path using UCB
        Q = []
        path_to_goal = []
        cost_path_to_goal = 0
        parent_map = {}
        heapq.heappush(Q, (0, start_point))
        cost_map = {(start_point[0], start_point[1]): 0}
        time_map = {(start_point[0], start_point[1]): 0}
        num_layers = len(sampled_points_list)
        # add to dict points layer the start
        dict_points_layer[(start_point[0], start_point[1])] = 0
        while Q:
            cost, current = heapq.heappop(Q)
            time_to_reach = time_map[(current[0], current[1])]
            if cost > cost_map[(current[0], current[1])]:
                continue
            if np.isclose(current, goal_point).all():
                # append to path_to_goal the spline between current and current's parent
                parent = parent_map[(current[0], current[1])]
                path_to_goal.append(all_discretized_splines_dict[(parent[0], parent[1], current[0], current[1])])
                cost_path_to_goal = cost
                time_path_to_goal = time_to_reach
                while not np.isclose(current, start_point).all():
                    parent = parent_map[(current[0], current[1])]
                    path_to_goal.insert(0, all_discretized_splines_dict[(parent[0], parent[1], current[0], current[1])])
                    current = parent
                if not np.isclose(path_to_goal[0][0], start_point).all():
                    path_to_goal.insert(
                        0,
                        all_discretized_splines_dict[
                            (start_point[0], start_point[1], path_to_goal[0][0][0], path_to_goal[0][0][1])
                        ],
                    )
                break
            if dict_points_layer[(current[0], current[1])] == num_layers and not np.isclose(current, start_point).all():
                # don't have other neighbors
                continue
            set_of_neighbors = sampled_points_list[dict_points_layer[(current[0], current[1])]]
            for i in range(len(set_of_neighbors)):
                dest_point = sampled_points_list[dict_points_layer[(current[0], current[1])]][i]
                spline_to_dest = all_discretized_splines_dict[(current[0], current[1], dest_point[0], dest_point[1])]
                time_to_reach_dest = time_to_reach + self.eval_time_to_reach(spline_to_dest, vx)
                obj = self.objective_function(spline_to_dest, vx, time_to_reach, lane_id)
                new_cost_to_reach = cost + obj
                cost_to_reach_neighbor = cost_map.get(
                    (dest_point[0], dest_point[1]), float("inf")
                )  # if not in the map, return inf
                if new_cost_to_reach < cost_to_reach_neighbor:
                    cost_map[(dest_point[0], dest_point[1])] = new_cost_to_reach
                    parent_map[(dest_point[0], dest_point[1])] = current
                    time_map[(dest_point[0], dest_point[1])] = time_to_reach_dest
                    heapq.heappush(Q, (new_cost_to_reach, dest_point))
        return path_to_goal, cost_path_to_goal, time_path_to_goal

    def eval_time_to_reach(self, spline, vx):
        time_to_reach = 0
        for i in range(1, len(spline)):
            distance = np.linalg.norm(np.array(spline[i]) - np.array(spline[i - 1]))
            time_to_reach += distance / vx
        return time_to_reach

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
                plt.plot(spline_x, spline_y, linewidth=4)
            else:
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

    def plot_path_and_cars(
        self, all_discretized_splines: List[List[Tuple[float, float]]], best_path: Lanelet, vx: float, t0=0
    ):
        plt.figure(figsize=(10, 6))

        # Plot all discretized splines
        for spline_points in all_discretized_splines:
            spline_x, spline_y = zip(*spline_points)
            plt.plot(spline_x, spline_y, label="Discretized Spline", color="gray", alpha=0.5)

        best_path_x, best_path_y = zip(*best_path.center_vertices)
        plt.plot(best_path_x, best_path_y, label="Best Path", color="blue", linewidth=2)
        # self.predict_ego_positions(best_path.center_vertices, vx)
        self.predict_other_cars_positions(best_path.center_vertices, vx, t0)
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
        closest_timestep = t0
        # Plot the polygons for the closest timestep
        for key, polygons in self.cars.items():
            poly = polygons[closest_timestep]
            x, y = poly.exterior.xy
            plt.plot(x, y, label=f"{key} Predicted Position", linewidth=1.5)
            plt.fill(x, y, alpha=0.4)

        x_vals = np.linspace(50, 100, 100)  # Adjust min_x and max_x as needed

        if self.sampling_on_goal_lane:
            lane_id = self.goal_lanelet_id
        else:
            lane_id = self.current_ego_lanelet_id
        a, b, c = self.center_lines[lane_id]

        # Calculate corresponding y values
        y_vals = (-a * x_vals - c) / b

        # Plot the line
        plt.plot(x_vals, y_vals, label="Line ax + by + c = 0", color="red", linewidth=2)

        plt.title("Best Path, and Predicted Car Positions at closest timestep")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.savefig("all_splines_best_path_and_cars.png")
        plt.close()

    def find_vx_same_lane(self):
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

    def find_vx_goal_lane(self):
        min_dist = float("inf")
        vx = self.sim_obs.players["Ego"].state.vx
        for keys in self.sim_obs.players:
            if keys != "Ego":
                lane_id = self.lanelet_network.find_lanelet_by_position(
                    [np.array([self.sim_obs.players[keys].state.x, self.sim_obs.players[keys].state.y])]
                )[0]
                if not lane_id:
                    # print("DISASTRO")
                    continue
                lane_id = lane_id[0]
                if lane_id == self.goal_lanelet_id:
                    r_to_car = (
                        np.array([self.sim_obs.players[keys].state.x, self.sim_obs.players[keys].state.y])
                        - self.ego_position
                    )
                    direction_player_lanelet = (
                        self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).center_vertices[1]
                        - self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).center_vertices[0]
                    ) / np.linalg.norm(
                        self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).center_vertices[1]
                        - self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).center_vertices[0]
                    )

                    signed_dist_to_car = np.dot(r_to_car, direction_player_lanelet)
                    if signed_dist_to_car < min_dist and signed_dist_to_car > 0:
                        min_dist = signed_dist_to_car
                        vx = self.sim_obs.players[keys].state.vx
        return vx

    def objective_function(self, spline: List[Tuple[float, float]], vx, t0, lane_id) -> float:
        objective_value = 0.0
        # t_predict = time.time()
        self.predict_other_cars_positions(spline, vx, t0)
        # print("Predict time: ", time.time() - t_predict)
        # t_collision = time.time()
        collisions = self.detect_collisions()
        for timestep in collisions:
            if collisions[timestep]:
                # print("COLLISION DETECTED", collisions[timestep])
                objective_value += 1000000
        # print("Collision time: ", time.time() - t_collision)
        # if any(collisions[timestep] for timestep in collisions):
        #     objective_value += 1000000
        for i, spli in enumerate(spline):
            # Smoothness term
            # TODO I'd like to do this analytically with the coefficients of the spline and evaluating the integrals but
            # an appropriate data structure would be needed
            # Guidance term
            a, b, c = self.center_lines[lane_id]
            distance = self.distance_point_to_line_2d(a, b, c, spline[i][0], spline[i][1])
            objective_value += (
                100
                * distance  # not super correct, should be an integral (as long as we don't compute intergrals is okay??)
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

    def compute_center_lines_coefficients(self, ego_position, lanes: List[Lanelet]):
        """
        Compute a dictionary of lanelet line coefficients (a, b, c) for each lanelet.

        Parameters:
        lanelet_network: An object containing lanelets, where each lanelet has an ID and center points.

        Returns:
        dict: A dictionary where keys are lanelet IDs and values are the coefficients (a, b, c) of the line
            passing through the first and last center points.
        """

        for lanelet in lanes:
            lanelet_id = lanelet.lanelet_id
            center_points = np.array(lanelet.center_vertices)

            # Extract the first and last points

            distance = np.linalg.norm(center_points - ego_position, axis=1)
            p1 = center_points[np.argmin(distance)]
            p2 = center_points[min(np.argmin(distance) + 4, len(center_points) - 1)]
            if (p1 == p2).all():
                continue

            # Compute the line coefficients
            # Line equation: ax + by + c = 0
            # a = y2 - y1,
            # b = x1 - x2
            # c = x2*y1 - x1*y2
            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = p2[0] * p1[1] - p1[0] * p2[1]

            # Store the coefficients in the dictionary
            self.center_lines[lanelet_id] = [a, b, c]

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

    def predict_ego_positions(self, spline, vx, t0) -> List[float]:
        """update cars dictionary and return timesteps list"""
        ego_vx = vx
        timesteps = [t0]
        a = spline[1][1] - spline[0][1]
        b = spline[0][0] - spline[1][0]
        m = -a / b
        theta = np.arctan(m)
        self.cars["Ego"] = [self.create_oriented_rectangle(spline[0], theta, "Ego")]
        if ego_vx < 1e-3:
            return timesteps
        for i in range(1, len(spline) - 1):
            distance = np.linalg.norm(
                np.array([spline[i][0], spline[i][1]]) - np.array([spline[i + 1][0], spline[i + 1][1]])
            )
            timesteps.append(distance / ego_vx + timesteps[-1])

            a = spline[i + 1][1] - spline[i][1]
            b = spline[i][0] - spline[i + 1][0]
            m = -a / b
            theta = np.arctan(m)
            self.cars["Ego"].append(self.create_oriented_rectangle(spline[i], theta, "Ego"))

        return timesteps

    def predict_other_cars_positions(self, spline, vx, t0) -> List[Polygon]:
        """
        Predict the future positions of static obstacles.

        Parameters:
        sim_obs (SimObservations): The current simulation observations.

        Returns:
        List[StaticObstacle]: A list of static obstacles with predicted positions.
        """

        self.cars = {}
        timesteps = self.predict_ego_positions(spline, vx, t0)
        for car in self.sim_obs.players:
            if car != "Ego":
                self.cars[car] = []
                x0 = self.sim_obs.players[car].state.x
                y0 = self.sim_obs.players[car].state.y
                r_to_car = (
                    np.array([self.sim_obs.players[car].state.x, self.sim_obs.players[car].state.y]) - self.ego_position
                )
                direction_player_lanelet = (
                    self.lanelet_network.find_lanelet_by_id(self.current_ego_lanelet_id).center_vertices[1]
                    - self.lanelet_network.find_lanelet_by_id(self.current_ego_lanelet_id).center_vertices[0]
                ) / np.linalg.norm(
                    self.lanelet_network.find_lanelet_by_id(self.current_ego_lanelet_id).center_vertices[1]
                    - self.lanelet_network.find_lanelet_by_id(self.current_ego_lanelet_id).center_vertices[0]
                )

                signed_dist_to_car = np.dot(r_to_car, direction_player_lanelet)
                if (
                    signed_dist_to_car < -1.5
                    and abs(self.sim_obs.players[car].state.vx - self.sim_obs.players["Ego"].state.vx) < 2
                    and self.traffic
                    and False
                ):
                    # if signed_dist_to_car < -1.5:
                    vx = 0
                    # print("Car is behind")
                else:
                    vx = self.sim_obs.players[car].state.vx
                psi = self.sim_obs.players[car].state.psi
                if signed_dist_to_car < -1.5:
                    # constant deceleration model
                    x = x0
                    y = y0
                    for t in timesteps:
                        if vx - 8 * t > 0:
                            added_space = vx * t - 0.5 * 8 * (t**2)
                            if added_space > 0:
                                x = x0 + added_space * np.cos(psi)
                                y = y0 + added_space * np.sin(psi)
                        self.cars[car].append(self.create_oriented_rectangle([x, y], psi, car))
                else:
                    for t in timesteps:
                        x = x0 + vx * t * np.cos(psi)
                        y = y0 + vx * t * np.sin(psi)
                        self.cars[car].append(self.create_oriented_rectangle([x, y], psi, car))

    def create_oriented_rectangle(self, center, theta, car_id):
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
        car = self.sim_obs.players[car_id]
        local_corners = np.array(car.occupancy.exterior.coords) - np.array(
            [self.sim_obs.players[car_id].state.x, self.sim_obs.players[car_id].state.y]
        ).reshape(1, -1)

        local_corners = local_corners * 1.1

        # Rotation matrix for the given orientation
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Rotate and translate the corners to global coordinates
        # global_corners = rotation_matrix @ local_corners.T + np.array(center).reshape(-1, 1)
        global_corners = local_corners.T + np.array(center).reshape(-1, 1)

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

    def set_traffic(self):
        self.traffic = True
