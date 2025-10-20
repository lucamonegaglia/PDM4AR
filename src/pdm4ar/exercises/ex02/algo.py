<<<<<<< HEAD
from abc import abstractmethod, ABC

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        Q = [start]
        V = [start]
        P = {start: None}
        opened_nodes = []
        while Q:
            s = Q.pop(0)
            opened_nodes.append(s)
            if s == goal:
                path = [s]
                while s != start:
                    s = P[s]
                    path = [s] + path
                return path, opened_nodes
            for v in sorted(graph[s], reverse=True):
                if v not in V:
                    Q = [v] + Q
                    V.append(v)
                    P[v] = s
        return [], opened_nodes


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        Q = [start]
        V = [start]
        P = {start: None}
        opened_nodes = []
        while Q:
            s = Q.pop(0)
            opened_nodes.append(s)
            if s == goal:
                path = [s]
                while s != start:
                    s = P[s]
                    path = [s] + path
                return path, opened_nodes
            for v in sorted(graph[s]):
                if v not in V:
                    Q.append(v)
                    V.append(v)
                    P[v] = s
        return [], opened_nodes


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        d = 1
        Q = [start]
        V = [start]
        P = {start: None}
        path_to = {start: [start]}
        opened_nodes = []
        max_depth = len(graph.keys())
        while d <= max_depth:
            while Q:
                s = Q.pop(0)
                opened_nodes.append(s)
                if s == goal:
                    path = [s]
                    while s != start:
                        s = P[s]
                        path = [s] + path
                    return path, opened_nodes
                for v in sorted(graph[s], reverse=True):
                    if v not in V and len(path_to[s]) < d:
                        Q = [v] + Q
                        V.append(v)
                        P[v] = s
                        path_to[v] = path_to[s] + [v]
            last_opened_nodes = opened_nodes.copy()
            opened_nodes = []
            V = [start]
            P = {start: None}
            Q = [start]
            path_to = {start: [start]}

            d += 1
        return [], last_opened_nodes
=======
from abc import abstractmethod, ABC

from pdm4ar.exercises.ex02.structures import AdjacencyList, X, Path, OpenedNodes


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, [] if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> tuple[Path, OpenedNodes]:
        # todo implement here your solution
        return [], []
>>>>>>> ex11/master
