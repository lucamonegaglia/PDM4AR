from abc import ABC, abstractmethod
from dataclasses import dataclass
import heapq  # you may find this helpful

from osmnx.distance import great_circle_vec

from pdm4ar.exercises.ex02.structures import X, Path
from pdm4ar.exercises.ex03.structures import WeightedGraph, TravelSpeed


@dataclass
class InformedGraphSearch(ABC):
    graph: WeightedGraph

    @abstractmethod
    def path(self, start: X, goal: X) -> Path:
        # Abstract function. Nothing to do here.
        pass


@dataclass
class UniformCostSearch(InformedGraphSearch):
    def path(self, start: X, goal: X) -> Path:
        # todo
        Q = []
        heapq.heappush(Q, (0, start))
        P = {start: None}
        costToReach = {start: 0}
        opened_nodes = []
        while Q:
            s = heapq.heappop(Q)[1]
            if s in opened_nodes:
                continue
            opened_nodes.append(s)
            if s == goal:
                path = [s]
                while s != start:
                    s = P[s]
                    path = [s] + path
                return path
            for v in self.graph.adj_list[s]:
                new_costToReach = costToReach[s] + self.graph.get_weight(s, v)
                if v not in costToReach or new_costToReach < costToReach[v]:
                    costToReach[v] = new_costToReach
                    heapq.heappush(Q, (new_costToReach, v))
                    P[v] = s
        return []


@dataclass
class Astar(InformedGraphSearch):

    # Keep track of how many times the heuristic is called
    heuristic_counter: int = 0
    # Allows the tester to switch between calling the students heuristic function and
    # the trivial heuristic (which always returns 0). This is a useful comparison to
    # judge how well your heuristic performs.
    use_trivial_heuristic: bool = False

    def heuristic(self, u: X, v: X) -> float:
        # Increment this counter every time the heuristic is called, to judge the performance
        # of the algorithm
        self.heuristic_counter += 1
        if self.use_trivial_heuristic:
            return 0
        else:
            # return the heuristic that the student implements
            return self._INTERNAL_heuristic(u, v)

    # Implement the following two functions

    def _INTERNAL_heuristic(self, u: X, v: X) -> float:
        # Implement your heuristic here. Your `path` function should NOT call
        # this function directly. Rather, it should call `heuristic`
        # todo
        u_lon, u_lat = self.graph.get_node_coordinates(u)
        v_lon, v_lat = self.graph.get_node_coordinates(v)

        euclid_distance = great_circle_vec(u_lat, u_lon, v_lat, v_lon)
        speed_to_other_nodes = 0
        for i in self.graph.adj_list[u]:
            i_lon, i_lat = self.graph.get_node_coordinates(i)
            speed_to_other_nodes = max(
                great_circle_vec(u_lat, u_lon, i_lat, i_lon) / self.graph.get_weight(u, i), speed_to_other_nodes
            )

        if speed_to_other_nodes > TravelSpeed.SECONDARY.value:
            speed = TravelSpeed.HIGHWAY.value
        else:
            speed = TravelSpeed.SECONDARY.value

        
        return euclid_distance / speed_to_other_nodes

        """ this made it worse   
        if speed_to_other_nodes > TravelSpeed.SECONDARY.value:
            speed = TravelSpeed.HIGHWAY.value
        elif speed_to_other_nodes > TravelSpeed.CITY.value:
            speed = TravelSpeed.SECONDARY.value
        elif speed_to_other_nodes > TravelSpeed.PEDESTRIAN.value:
            speed = TravelSpeed.CITY.value
        else:
            speed = TravelSpeed.PEDESTRIAN.value
        return euclid_distance / speed
        """

    def path(self, start: X, goal: X) -> Path:
        # todo
        Q = []
        heapq.heappush(Q, (0 + self.heuristic(start, goal), start))
        P = {start: None}
        costToReach = {start: 0.0}
        opened_nodes = []
        while Q:
            s = heapq.heappop(Q)[1]
            if s in opened_nodes:
                continue
            opened_nodes.append(s)
            if s == goal:
                path = [s]
                while s != start:
                    s = P[s]
                    path = [s] + path
                return path
            for v in self.graph.adj_list[s]:
                new_costToReach = costToReach[s] + self.graph.get_weight(s, v)
                if v not in costToReach or new_costToReach < costToReach[v]:
                    costToReach[v] = new_costToReach
                    heapq.heappush(Q, (new_costToReach + self.heuristic(v, goal), v))
                    P[v] = s
        return []


def compute_path_cost(wG: WeightedGraph, path: Path):
    """A utility function to compute the cumulative cost along a path"""
    if not path:
        return float("inf")
    total: float = 0
    for i in range(1, len(path)):
        inc = wG.get_weight(path[i - 1], path[i])
        total += inc
    return total
