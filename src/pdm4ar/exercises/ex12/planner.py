from mimetypes import init
import random
from dataclasses import dataclass
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

@dataclass
class Planner:
    lanelet_network: LaneletNetwork
    player_name: PlayerName
    planning_goal: PlanningGoal

    def sample_points_on_lane(self, lane_id: int, num_points: int) -> Sequence[np.ndarray]:
        lanelet = self.lanelet_network.find_lanelet_by_id(lane_id)
        if not lanelet:
            raise ValueError(f"Lanelet with id {lane_id} not found")
        
        sampled_points = []
        for _ in range(num_points):
            s = random.uniform(0, lanelet.length)
            points = lanelet.interpolate_position(s)    #The interpolated positions on the center/right/left polyline and the segment id of the polyline where the interpolation takes place in the form ([x_c,y_c],[x_r,y_r],[x_l,y_l], segment_id)
            sampled_points.append(points)
        
        return sampled_points

    def sample_points_on_player_lane(self, num_points: int) -> Sequence[np.ndarray]:
        player_lane_id = self.get_player_lane_id()
        return self.sample_points_on_lane(player_lane_id, num_points)

    def sample_points_on_goal_lane(self, num_points: int) -> Sequence[np.ndarray]:
        goal_lane_id = self.get_goal_lane_id()
        return self.sample_points_on_lane(goal_lane_id, num_points)

    def get_player_lane_id(self) -> int:
        # Placeholder for actual implementation to get player's lane id
        pass

    def get_goal_lane_id(self) -> int:
        # Placeholder for actual implementation to get goal's lane id
        pass