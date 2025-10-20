from email.utils import collapse_rfc2231_value
from tabnanny import check
from typing import List, Union
from dg_commons import SE2Transform
from pyparsing import col
from pdm4ar.exercises.ex06.collision_primitives import (
    CollisionPrimitives,
    CollisionPrimitives_SeparateAxis,
)
from pdm4ar.exercises_def.ex06.structures import (
    Polygon,
    GeoPrimitive,
    Point,
    Segment,
    Circle,
    Triangle,
    Path,
)
import shapely
import numpy as np

##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

COLLISION_PRIMITIVES = {
    Point: {
        Circle: lambda x, y: CollisionPrimitives.circle_point_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_point_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_point_collision(y, x),
    },
    Segment: {
        Circle: lambda x, y: CollisionPrimitives.circle_segment_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_segment_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_segment_collision_aabb(y, x),
    },
    Triangle: {
        Point: CollisionPrimitives.triangle_point_collision,
        Segment: CollisionPrimitives.triangle_segment_collision,
        Polygon: CollisionPrimitives.triangle_polygon_collision,
        Circle: lambda x, y: CollisionPrimitives.circle_triangle_collision(y, x),
        Triangle: CollisionPrimitives.triangle_triangle_collision,
    },
    Circle: {
        Point: CollisionPrimitives.circle_point_collision,
        Segment: CollisionPrimitives.circle_segment_collision,
        Circle: CollisionPrimitives.circle_circle_collision,
        Polygon: CollisionPrimitives.circle_polygon_collision,
        Triangle: CollisionPrimitives.circle_triangle_collision,
    },
    Polygon: {
        Point: CollisionPrimitives.polygon_point_collision,
        Segment: CollisionPrimitives.polygon_segment_collision_aabb,
        Circle: lambda x, y: CollisionPrimitives.circle_polygon_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_polygon_collision(y, x),
        Polygon: CollisionPrimitives.polygon_polygon_collision,
    },
}


def check_collision(p_1: GeoPrimitive, p_2: GeoPrimitive) -> bool:
    """
    Checks collision between 2 geometric primitives
    Note that this function only uses the functions that you implemented in CollisionPrimitives class.
        Parameters:
                p_1 (GeoPrimitive): Geometric Primitive
                p_w (GeoPrimitive): Geometric Primitive
    """
    assert type(p_1) in COLLISION_PRIMITIVES, "Collision primitive does not exist."
    assert type(p_2) in COLLISION_PRIMITIVES[type(p_1)], "Collision primitive does not exist."

    collision_func = COLLISION_PRIMITIVES[type(p_1)][type(p_2)]

    return collision_func(p_1, p_2)


##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not


def geo_primitive_to_shapely(p: GeoPrimitive):
    """
    Given function.

    Casts a geometric primitive into a Shapely object. Feel free to use this function or not
    for the later tasks.
    """
    if isinstance(p, Point):
        return shapely.Point(p.x, p.y)
    elif isinstance(p, Segment):
        return shapely.LineString([[p.p1.x, p.p1.y], [p.p2.x, p.p2.y]])
    elif isinstance(p, Circle):
        return shapely.Point(p.center.x, p.center.y).buffer(p.radius)
    elif isinstance(p, Triangle):
        return shapely.Polygon([[p.v1.x, p.v1.y], [p.v2.x, p.v2.y], [p.v3.x, p.v3.y]])
    else:  # Polygon
        vertices = []
        for vertex in p.vertices:
            vertices += [(vertex.x, vertex.y)]
        return shapely.Polygon(vertices)


class CollisionChecker:
    """
    This class implements the collision check ability of a simple planner for a circular differential drive robot.

    Note that check_collision could be used to check collision between given GeoPrimitives
    check_collision function uses the functions that you implemented in CollisionPrimitives class.
    """

    def __init__(self):
        pass

    def path_collision_check(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (list[GeoPrimitive]): list of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        collision_indices = []
        for i, p in enumerate(t.waypoints[:-1]):
            seg = Segment(p, t.waypoints[i + 1])
            occupancy_shapes = self.create_occupancy_shape(seg, r)
            for o in obstacles:
                for s in occupancy_shapes:
                    if check_collision(s, o):
                        collision_indices.append(i)
                        break
                else:
                    continue  # Only executed if the inner loop didn't break
                break
        return collision_indices

    def create_occupancy_shape(self, t: Union[Path, Segment], r: float) -> list[Polygon]:
        """
        Returns the occupancy shape of the given path and radius of the circular differential drive robot.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
        """
        if isinstance(t, Segment):
            path = Path([t.p1, t.p2])
        else:
            path = t
        occupancy_shapes = []
        for i, s in enumerate(path.waypoints[:-1]):
            segment_start = s
            segment_end = path.waypoints[i + 1]
            occupancy_shapes.append(Circle(segment_start, r))
            occupancy_shapes.append(Circle(segment_end, r))

            segment_direction = np.array([segment_end.x - segment_start.x, segment_end.y - segment_start.y])
            segment_direction = segment_direction / np.linalg.norm(segment_direction)
            segment_normal = np.array([-segment_direction[1], segment_direction[0]])
            vertex_1 = Point(segment_start.x + r * segment_normal[0], segment_start.y + r * segment_normal[1])
            vertex_2 = Point(segment_start.x - r * segment_normal[0], segment_start.y - r * segment_normal[1])
            vertex_3 = Point(segment_end.x - r * segment_normal[0], segment_end.y - r * segment_normal[1])
            vertex_4 = Point(segment_end.x + r * segment_normal[0], segment_end.y + r * segment_normal[1])

            polygon = Polygon([vertex_1, vertex_2, vertex_3, vertex_4])
            occupancy_shapes.append(polygon)
        return occupancy_shapes

    def path_collision_check_occupancy_grid(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will generate an occupancy grid of the given map.
        Then, occupancy grid will be used to check collisions.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (list[GeoPrimitive]): list of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        xmin = min([p.x for p in t.waypoints]) - r
        xmax = max([p.x for p in t.waypoints]) + r
        ymin = min([p.y for p in t.waypoints]) - r
        ymax = max([p.y for p in t.waypoints]) + r

        xdim = xmax - xmin
        ydim = ymax - ymin
        step = max(xdim, ydim) / 100

        obstacle_grid = np.zeros((int(ydim / step), int(xdim / step)), dtype=bool)
        # robot_grid = np.zeros_like(obstacle_grid, dtype=bool)
        l = []
        # occupancy_grid = np.zeros_like(grid, dtype=bool)
        for o in obstacles:
            pmin, pmax = o.get_boundaries()
            for i in range(int(pmin.x / step), int(pmax.x / step) + 1):
                for j in range(int(pmin.y / step), int(pmax.y / step) + 1):
                    if 0 <= j < obstacle_grid.shape[0] and 0 <= i < obstacle_grid.shape[1]:
                        obstacle_grid[j, i] = check_collision(Point(i * step, j * step), o)

        for segment_idx, start in enumerate(t.waypoints[:-1]):
            seg = Segment(start, t.waypoints[segment_idx + 1])
            occupancy_shapes = self.create_occupancy_shape(seg, r)
            for o in occupancy_shapes:
                pmin, pmax = o.get_boundaries()
                for i in range(int(pmin.x / step), int(pmax.x / step) + 1):
                    for j in range(int(pmin.y / step), int(pmax.y / step) + 1):
                        if check_collision(Point(i * step, j * step), o) and obstacle_grid[j, i]:
                            l.append(segment_idx)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

        """for segment_idx, start in enumerate(t.waypoints[:-1]):
            seg = Segment(start, t.waypoints[segment_idx + 1])
            segment_samples = CollisionPrimitives.sample_segment(seg)
            for p in segment_samples:
                i, j = int(p.x / step), int(p.y / step)
                if 0 <= j < grid.shape[0] and 0 <= i < grid.shape[1]:
                    if grid[j, i]:
                        l.append(segment_idx)
                        break
        """
        return l

    def path_collision_check_r_tree(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will build an R-Tree of the given obstacles.
        You are free to implement your own R-Tree or you could use STRTree of shapely module.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        shapely_obstacles = [geo_primitive_to_shapely(o) for o in obstacles]
        tree = shapely.STRtree(shapely_obstacles)
        l = []
        for i, p in enumerate(t.waypoints[:-1]):
            seg = Segment(p, t.waypoints[i + 1])
            for s in self.create_occupancy_shape(seg, r):
                intersecting_bbs = tree.query(geo_primitive_to_shapely(s))

                for j in intersecting_bbs:
                    if check_collision(s, obstacles[j]):
                        l.append(i)
                        break
                else:
                    continue
                break
        return l

    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: SE2Transform,
        next_pose: SE2Transform,
        observed_obstacles: list[GeoPrimitive],
    ) -> bool:
        """
        Returns there exists a collision or not during the movement of a circular differential drive robot until its next pose.

            Parameters:
                    r (float): Radius of circular differential drive robot
                    current_pose (SE2Transform): Current pose of the circular differential drive robot
                    next_pose (SE2Transform): Next pose of the circular differential drive robot
                    observed_obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives in robot frame
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        v = np.array([next_pose.p[0] - current_pose.p[0], next_pose.p[1] - current_pose.p[1]])
        rotated_v = np.array(
            [
                v[0] * np.cos(current_pose.theta) + v[1] * np.sin(current_pose.theta),
                -v[0] * np.sin(current_pose.theta) + v[1] * np.cos(current_pose.theta),
            ]
        )
        seg = Segment(Point(0, 0), Point(rotated_v[0], rotated_v[1]))
        for s in self.create_occupancy_shape(seg, r):
            for o in observed_obstacles:
                if check_collision(s, o):
                    return True
        return False

    def path_collision_check_safety_certificate(self, t: Path, r: float, obstacles: list[GeoPrimitive]) -> list[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will implement the safety certificates procedure for collision checking.
        You are free to use shapely to calculate distance between a point and a GeoPrimitive.
        For more information, please check Algorithm 1 inside the following paper:
        https://journals.sagepub.com/doi/full/10.1177/0278364915625345.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (list[GeoPrimitive]): list of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        self.Sfree: list[Point] = []
        self.Sobs: list[Point] = []
        self.CertifierOf: dict[Point, Point] = {}
        self.dist: dict[Point, float] = {}
        l = []

        xmin_path = min([p.x for p in t.waypoints]) - r
        xmax_path = max([p.x for p in t.waypoints]) + r
        ymin_path = min([p.y for p in t.waypoints]) - r
        ymax_path = max([p.y for p in t.waypoints]) + r

        xmin_obs = min([o.get_boundaries()[0].x for o in obstacles]) - 1
        xmax_obs = max([o.get_boundaries()[1].x for o in obstacles]) + 1
        ymin_obs = min([o.get_boundaries()[0].y for o in obstacles]) - 1
        ymax_obs = max([o.get_boundaries()[1].y for o in obstacles]) + 1

        xmin = min(xmin_path, xmin_obs)
        xmax = max(xmax_path, xmax_obs)
        ymin = min(ymin_path, ymin_obs)
        ymax = max(ymax_path, ymax_obs)

        bounding_area = shapely.geometry.Polygon(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
        )  # Adjust as needed
        self.free_space = bounding_area
        # Assume `obstacles` is a list of Geoprimitives
        # Convert each obstacle to a Shapely Polygon (or appropriate shape)
        shapely_obstacles = [geo_primitive_to_shapely(o).buffer(0) for o in obstacles]

        self.obs_space = shapely.geometry.Polygon()
        for obstacle in shapely_obstacles:
            self.obs_space = self.obs_space.union(obstacle)
            self.free_space = self.free_space.difference(obstacle)

        for i, p_start in enumerate(t.waypoints[:-1]):
            seg = Segment(p_start, t.waypoints[i + 1])
            # occupancy_shapes = self.create_occupancy_shape(seg, r)
            seg_lenght = np.linalg.norm([seg.p1.x - seg.p2.x, seg.p1.y - seg.p2.y])
            self.CFreePoint(p_start, r)
            if self.Dist(p_start) > seg_lenght + r:
                continue

            else:  # sample points on the segment blablabla

                for p_seg in CollisionPrimitives.sample_segment(seg):

                    if not self.CFreePoint(p_seg, r):
                        l.append(i)
                        break
                    partial_seg_lenght = np.linalg.norm(
                        [(p_seg.x, p_seg.y), (t.waypoints[i + 1].x, t.waypoints[i + 1].y)]
                    )  # remaining segment
                    if self.Dist(p_seg) > partial_seg_lenght + r:
                        break
        return l

    def CFreePoint(self, p: Point, r: float) -> bool:
        xfree = self.Nearest(p, self.Sfree)
        xobs = self.Nearest(p, self.Sobs)

        if xfree is not None and xobs is not None:
            if (
                np.linalg.norm([p.x - xfree.x, p.y - xfree.y]) < self.Dist(xfree) - r
            ):  # inside safety certificate of xfree
                self.CertifierOf[p] = xfree
                return True
            if (
                np.linalg.norm([p.x - xobs.x, p.y - xobs.y]) < self.Dist(xobs) + r
            ):  # inside (un)safety certificate of xobs
                return False

        dobs = self.SetDistance(self.obs_space, p)
        if dobs > r:
            self.Sfree.append(p)
            self.dist[p] = dobs
            self.CertifierOf[p] = p
            return True
        else:
            self.Sobs.append(p)
            self.dist[p] = self.SetDistance(self.free_space, p)
            return False

    def Nearest(self, p: Point, S: list[Point]) -> Union[Point, None]:
        min_distance = np.inf
        nearest = None
        for s in S:
            distance = np.linalg.norm([p.x - s.x, p.y - s.y])
            if distance < min_distance:
                min_distance = distance
                nearest = s
        return nearest

    def Dist(self, p: Point) -> float:
        if p in self.dist.keys():
            return self.dist[p]
        if p in self.Sobs:
            d = shapely.distance(geo_primitive_to_shapely(p), self.free_space)
            self.dist[p] = d
            return d
        elif p in self.Sfree:
            d = shapely.distance(geo_primitive_to_shapely(p), self.obs_space)
            self.dist[p] = d
            return d
        else:
            return 0

    def SetDistance(self, obstacles, p: Point) -> float:
        alpha = 1  # idk?
        return alpha * shapely.distance(geo_primitive_to_shapely(p), obstacles)
