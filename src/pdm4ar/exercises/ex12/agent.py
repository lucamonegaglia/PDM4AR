from mimetypes import init
import random
from dataclasses import dataclass
from tracemalloc import start
from typing import Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
import numpy as np
from pdm4ar.exercises.ex12.planner import Planner
from pdm4ar.exercises.ex12.controller import PurePursuitController
from pdm4ar.exercises_def.ex09 import goal
from shapely.geometry import Point
import time

import cProfile
import pstats
import io


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    name: PlayerName
    goal: PlanningGoal
    sg: VehicleGeometry
    sp: VehicleParameters
    start = True
    lanelet_network: LaneletNetwork
    control_points: Sequence
    n_unsuccessful_merging = 0

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = Planner()
        self.mycontroller = PurePursuitController()

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator only at the beginning of each simulation.
        Do not modify the signature of this method."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.sg = init_obs.model_geometry
        self.sp = init_obs.model_params
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.control_points = init_obs.goal.ref_lane.control_points
        self.cycle_counter = 0
        # print("Control points: ", self.control_points)

        self.goal_lanelet_id = self.lanelet_network.find_lanelet_by_position([self.control_points[1].q.p])[0][0]
        print("Goal lanelet id: ", self.goal_lanelet_id)
        print("Real goal lanelet", self.lanelet_network.find_lanelet_by_position([self.goal.goal_polygon.centroid]))
        self.arrived_at_goal = False
        self.predecessor_goal_lane_id = self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).predecessor
        if not self.predecessor_goal_lane_id:
            self.predecessor_goal_lane_id = None
        else:
            self.predecessor_goal_lane_id = self.predecessor_goal_lane_id[0]
            merged_lanelet = self.lanelet_network.find_lanelet_by_id(self.predecessor_goal_lane_id).merge_lanelets(
                self.lanelet_network.find_lanelet_by_id(self.predecessor_goal_lane_id),
                self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id),
            )
            self.lanelet_network.remove_lanelet(self.predecessor_goal_lane_id)
            self.lanelet_network.remove_lanelet(self.goal_lanelet_id)
            self.lanelet_network.add_lanelet(merged_lanelet)
            self.goal_lanelet_id = merged_lanelet.lanelet_id
        self.myplanner.initialize(self.lanelet_network, self.name, self.goal, self.sg, self.goal_lanelet_id)

        # print(init_obs.dg_scenario.lanelet_network)
        # print(init_obs.dg_scenario.lanelet_network.find_lanelet_by_position())

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """This method is called by the simulator every dt_commands seconds (0.1s by default).
        Do not modify the signature of this method.

        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state
        :param sim_obs:
        :return:
        """

        if self.start:
            ego_position = np.array([sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y])
            self.myplanner.compute_center_lines_coefficients(ego_position)
            # calculate average velocity of the cars in the goal lane
            tot_speed = 0
            n_cars_goal_lane = 0
            for car in sim_obs.players:
                if car != "Ego":
                    position = np.array([sim_obs.players[car].state.x, sim_obs.players[car].state.y])
                    if self.lanelet_network.find_lanelet_by_position([position])[0][0] == self.goal_lanelet_id:
                        tot_speed += sim_obs.players[car].state.vx
                        n_cars_goal_lane += 1
            # max_distance = 0
            # # calculate the density of the cars in the goal lane
            # if n_cars_goal_lane > 0:
            #     # calculate density using goal lanelet length
            #     density = self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id).distance[-1] / n_cars_goal_lane
            #     print(f"Density in the goal lane: {density}")
            if n_cars_goal_lane > 0:
                avg_speed = tot_speed / n_cars_goal_lane
                if avg_speed < 1.7:
                    self.myplanner.set_traffic()
                    print(f"TRAFFIC, average speed: {avg_speed}")
                else:
                    print(f"No traffic, average speed: {avg_speed}")

            self.start = False
        if self.cycle_counter % 3 == 0:
            # self.flag = False
            # profiler = cProfile.Profile()
            # profiler.enable()  # Avvia il profiler
            # t = time.time()
            ego_position = np.array([sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y])
            l = self.lanelet_network.find_lanelet_by_position([ego_position])[0]
            if not l:
                print("Outside of lane")
                return VehicleCommands(acc=0, ddelta=0)
            current_ego_lanelet_id = l[0]

            ego_lanelet = self.lanelet_network.find_lanelet_by_id(current_ego_lanelet_id)
            goal_lanelet = self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id)

            # vector of goal wrt to position and normalized direction of player lanelet, used to determine if goal is in front of us or behind us
            r_to_goal_lanelet = -goal_lanelet.center_vertices[0] + ego_position
            direction_player_lanelet = (
                ego_lanelet.center_vertices[1] - ego_lanelet.center_vertices[0]
            ) / np.linalg.norm(ego_lanelet.center_vertices[1] - ego_lanelet.center_vertices[0])

            signed_dist_to_goal_lane = np.dot(r_to_goal_lanelet, direction_player_lanelet)
            # dist_to_goal = self.goal.goal_polygon.distance(Point(ego_position))
            # if dist_to_goal < 1 or self.arrived_at_goal:
            center_vertices = goal_lanelet.center_vertices
            distance = -center_vertices[0] + ego_position
            direction_player_lanelet = (
                goal_lanelet.center_vertices[-1] - goal_lanelet.center_vertices[0]
            ) / np.linalg.norm(goal_lanelet.center_vertices[-1] - goal_lanelet.center_vertices[0])

            projected_distance = np.dot(distance, direction_player_lanelet)

            if np.max(goal_lanelet.distance) - projected_distance < 1 or self.arrived_at_goal:
                self.arrived_at_goal = True
                print("Close to goal, sampling stopped")
                return VehicleCommands(acc=0, ddelta=0)

            self.myplanner.update_sim_obs(sim_obs)

            sampled_points_player_lane, dict_points_layer_player = self.myplanner.sample_points_on_lane_good(
                lane_id=current_ego_lanelet_id, num_points=4
            )

            if not sampled_points_player_lane:
                print("No sampled points")
                return VehicleCommands(acc=0, ddelta=0)

            all_splines_player_lane, all_splines_player_lane_dict = self.myplanner.get_all_discretized_splines(
                sampled_points_player_lane
            )
            if not all_splines_player_lane[0]:
                print("No splines on player lane")
                return VehicleCommands(acc=0, ddelta=0)

            all_splines_dict = {}
            all_splines_dict.update(all_splines_player_lane_dict)
            sample_points = sampled_points_player_lane
            dict_points_layer = {}
            dict_points_layer.update(dict_points_layer_player)

            # start position of the ego vehicle
            start_position = np.array([sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y])
            # end position at the center of the lane after the last layer of sampled points
            end_position = np.array(
                [
                    sampled_points_player_lane[-1][0][0],
                    sampled_points_player_lane[-1][0][1],
                ]
            )
            # do graph search for best path

            if (
                current_ego_lanelet_id != self.goal_lanelet_id
                and signed_dist_to_goal_lane >= -sim_obs.players["Ego"].state.vx
            ):
                sampled_points_goal_lane, dict_points_layer_goal = self.myplanner.sample_points_on_lane_good(
                    lane_id=self.goal_lanelet_id, num_points=4
                )
                if not sampled_points_goal_lane:
                    print("No sampled points")
                    return VehicleCommands(acc=0, ddelta=0)
                dict_points_layer.update(dict_points_layer_goal)
                for i in range(len(sample_points)):
                    sample_points[i] = sample_points[i] + sampled_points_goal_lane[i]
                # update end_position
                end_position = np.array(
                    [
                        sampled_points_goal_lane[-1][0][0],
                        sampled_points_goal_lane[-1][0][1],
                    ]
                )

                all_splines_goal_lane, all_splines_goal_lane_dict = self.myplanner.get_all_discretized_splines(
                    sample_points
                )
                if not all_splines_goal_lane[0]:
                    print("No splines on goal lane")
                    return VehicleCommands(acc=0, ddelta=0)
                all_splines_dict.update(all_splines_goal_lane_dict)
                all_splines = all_splines_goal_lane
                self.myplanner.set_sampling_on_goal_lane()
            else:
                all_splines_goal_lane = []
                all_splines = all_splines_player_lane
            self.myplanner.where_is_goal()
            # access vx of ego
            best_vx = 0
            if current_ego_lanelet_id == self.goal_lanelet_id:
                vx_car_ahead = self.myplanner.find_vx_same_lane()
                print("I am in the goal lane and follow the speed of the car ahead")
                best_vx = vx_car_ahead
                best_path, best_cost = self.myplanner.graph_search(
                    all_splines_dict,
                    sample_points,
                    dict_points_layer,
                    start_position,
                    end_position,
                    vx_car_ahead,
                )
            else:
                vx_vector = []
                vx_vector.append(sim_obs.players["Ego"].state.vx)
                vx_vector.append(self.myplanner.find_vx_same_lane())
                vx_vector.append(self.myplanner.find_vx_goal_lane())
                vx_vector_sorted = vx_vector

                best_vx = 0
                best_path = []
                # iniitial cost infinite
                best_cost = np.inf
                for o, v in enumerate(vx_vector_sorted):
                    path, cost = self.myplanner.graph_search(
                        all_splines_dict, sample_points, dict_points_layer, start_position, end_position, v
                    )
                    if cost < best_cost:
                        best_path = path
                        best_vx = v
                        best_cost = cost

                if best_cost > 100000 and current_ego_lanelet_id != self.goal_lanelet_id:
                    if self.n_unsuccessful_merging >= 10:
                        vx_unsuccessful_merging = min(vx_vector[1], vx_vector[2] * 0.5)
                    else:
                        vx_unsuccessful_merging = vx_vector[1]
                    end_position = np.array(
                        [
                            sampled_points_player_lane[-1][1][0],
                            sampled_points_player_lane[-1][1][1],
                        ]
                    )
                    best_path, best_cost = self.myplanner.graph_search(
                        all_splines_dict,
                        sample_points,
                        dict_points_layer,
                        start_position,
                        end_position,
                        vx_unsuccessful_merging,
                    )
                    best_vx = vx_unsuccessful_merging
                    self.n_unsuccessful_merging += 1

            # print best cost
            print("Best cost: ", best_cost)
            best_path = best_path[:-1]

            path = self.myplanner.merge_adjacent_splines(best_path)
            path = self.myplanner.get_path_from_waypoints(path)
            # self.myplanner.plot_path_and_cars(all_splines, path, best_vx, 0)
            collisions = self.myplanner.detect_collisions()
            for timestep in collisions:
                if collisions[timestep]:
                    print("COLLISION DETECTED", collisions[timestep])
                    break
            self.mycontroller.initialize(path)
            self.mycontroller.update_speed_reference(best_vx)
            # print("UPDATED SPEED REF TO", vx)
        self.cycle_counter += 1

        # rnd_acc = random.random() * self.params.param1
        # rnd_ddelta = (0) * self.params.param1
        # return VehicleCommands(acc=0.1, ddelta=0)
        # print(f"Speed: {sim_obs.players['Ego'].state.vx}")
        return self.mycontroller.compute_control(sim_obs.players["Ego"].state, float(sim_obs.time))
