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
    flag = True
    lanelet_network: LaneletNetwork
    control_points: Sequence

    def __init__(self):
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.myplanner = ()
        self.mycontroller = ()

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
        self.myplanner = Planner(self.lanelet_network, self.name, self.goal, self.sg, self.goal_lanelet_id)
        self.arrived_at_goal = False
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

        # todo implement here some better planning

        # In here you can find the observation of the lidar + the state of my vehicle
        # print(sim_obs.players)
        # print(sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y)

        # print("Current ego lanelet: ", current_ego_lanelet_id)

        if self.flag:
            ego_position = np.array([sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y])
            self.myplanner.compute_center_lines_coefficients(ego_position)
            self.flag = False
        if self.cycle_counter % 3 == 0:
            # self.flag = False
            # profiler = cProfile.Profile()
            # profiler.enable()  # Avvia il profiler
            # t = time.time()
            ego_position = np.array([sim_obs.players["Ego"].state.x, sim_obs.players["Ego"].state.y])
            # print("Position: ", ego_position)
            # print("vx: ", sim_obs.players["Ego"].state.vx)
            l = self.lanelet_network.find_lanelet_by_position([ego_position])[0]
            if not l:
                print("Outside of lane")
                return VehicleCommands(acc=0, ddelta=0)
            current_ego_lanelet_id = l[0]

            ego_lanelet = self.lanelet_network.find_lanelet_by_id(current_ego_lanelet_id)
            goal_lanelet = self.lanelet_network.find_lanelet_by_id(self.goal_lanelet_id)

            # vector of goal wrt to position and normalized direction of player lanelet, used to determine if goal is in front of us or behind us
            r_to_goal_lanelet = goal_lanelet.center_vertices[0] - ego_position
            direction_player_lanelet = (
                ego_lanelet.center_vertices[1] - ego_lanelet.center_vertices[0]
            ) / np.linalg.norm(ego_lanelet.center_vertices[1] - ego_lanelet.center_vertices[0])

            signed_dist_to_goal_lane = np.dot(r_to_goal_lanelet, direction_player_lanelet)
            dist_to_goal = self.goal.goal_polygon.distance(Point(ego_position))
            if dist_to_goal < 1 or self.arrived_at_goal:
                self.arrived_at_goal = True
                print("Close to goal, sampling stopped")
                return VehicleCommands(acc=0, ddelta=0)

            # print("Ego Lanelet Points", ego_lanelet.center_vertices)
            # print("Goal Lanelet Points", goal_lanelet.center_vertices)
            # self.flag = False
            # self.myplanner.plot_sampled_points(
            #     self.myplanner.sample_points_on_lane(lane_id=current_ego_lanelet_id, num_points=3),
            #     current_ego_lanelet_id,
            #     self.myplanner.get_discretized_spline(
            #         self.myplanner.sample_points_on_lane(lane_id=current_ego_lanelet_id, num_points=3)
            #     ),
            # )

            self.myplanner.update_sim_obs(sim_obs)

            sampled_points_player_lane, index_init_player, index_end_player, dict_points_layer_player = (
                self.myplanner.sample_points_on_lane(lane_id=current_ego_lanelet_id, num_points=4)
            )
            # print("Sampled points player lane: ", sampled_points_player_lane)
            if not sampled_points_player_lane:
                print("No sampled points")
                return VehicleCommands(acc=0, ddelta=0)
            bc_value_init = np.arctan2(
                sampled_points_player_lane[1][0][1] - sampled_points_player_lane[0][0][1],
                sampled_points_player_lane[1][0][0] - sampled_points_player_lane[0][0][0],
            )
            bc_value_end = np.arctan2(
                sampled_points_player_lane[-1][0][1] - sampled_points_player_lane[-2][0][1],
                sampled_points_player_lane[-1][0][0] - sampled_points_player_lane[-2][0][0],
            )
            # bc_value_init = self.control_points[index_init_player].q.theta
            # bc_value_end = self.control_points[index_end_player].q.theta
            all_splines_player_lane, all_splines_player_lane_dict = self.myplanner.get_all_discretized_splines(
                sampled_points_player_lane, bc_value_init, bc_value_end
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
                and signed_dist_to_goal_lane <= sim_obs.players["Ego"].state.vx
            ):
                sampled_points_goal_lane, index_init_goal, index_end_goal, dict_points_layer_goal = (
                    self.myplanner.sample_points_on_lane(lane_id=self.goal_lanelet_id, num_points=4)
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
                # print("Sampled points goal lane: ", sampled_points_goal_lane)
                bc_value_init = self.control_points[index_init_goal].q.theta
                bc_value_end = self.control_points[index_end_goal].q.theta

                all_splines_goal_lane, all_splines_goal_lane_dict = self.myplanner.get_all_discretized_splines(
                    sample_points, bc_value_init, bc_value_end
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
            vx = sim_obs.players["Ego"].state.vx
            best_vx = vx
            best_path1, cost1 = self.myplanner.graph_search(
                all_splines_dict,
                sample_points,
                dict_points_layer,
                start_position,
                end_position,
                vx,
            )
            vx2 = self.myplanner.find_vx()
            best_path2, cost2 = self.myplanner.graph_search(
                all_splines_dict,
                sample_points,
                dict_points_layer,
                start_position,
                end_position,
                vx2,
            )
            if cost1 < cost2:
                best_path = best_path1
                best_vx = vx
            else:
                best_path = best_path2
                best_vx = vx2
            best_path = best_path[:-1]
            # all_splines = all_splines_player_lane + all_splines_goal_lane

            # print("Number of sampled points: ", len(sampled_points_player_lane[0]) * len(sampled_points_player_lane))
            # print(sampled_points_player_lane[0][0][0], sampled_points_player_lane[0][0][1])
            # print(sampled_points_player_lane[0][1][0], sampled_points_player_lane[0][1][1])
            # print(sampled_points_player_lane[0][2][0], sampled_points_player_lane[0][2][1])

            self.myplanner.plot_all_discretized_splines(all_splines, best_path)

            # best_path_unified = []
            # for i in range(len(best_path)):
            #     best_path_unified += best_path[i]

            path = self.myplanner.merge_adjacent_splines(best_path)
            path = self.myplanner.get_path_from_waypoints(path)
            self.myplanner.plot_path_and_cars(all_splines, path, vx)
            collisions = self.myplanner.detect_collisions()
            for timestep in collisions:
                if collisions[timestep]:
                    print("COLLISION DETECTED", collisions[timestep])
                    break
            # path, vx = self.myplanner.get_best_path(all_splines)
            # self.myplanner.predict_other_cars_positions(path)
            # self.myplanner.plot_all_discretized_splines(all_splines, path.center_vertices)
            self.mycontroller = PurePursuitController(path)
            self.mycontroller.update_speed_reference(best_vx)
            # print("UPDATED SPEED REF TO", vx)
        self.cycle_counter += 1

        # rnd_acc = random.random() * self.params.param1
        # rnd_ddelta = (0) * self.params.param1
        # return VehicleCommands(acc=0.1, ddelta=0)
        # print(f"Speed: {sim_obs.players['Ego'].state.vx}")
        return self.mycontroller.compute_control(sim_obs.players["Ego"].state, float(sim_obs.time))
