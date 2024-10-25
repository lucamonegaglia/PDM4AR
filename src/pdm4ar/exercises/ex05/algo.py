from collections.abc import Sequence
from logging import config
from tracemalloc import start
from turtle import circle, forward

from dg_commons import SE2Transform
from numpy import short

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> list[SE2Transform]:
        """Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a list[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # TODO implement here your solution
    rmin = wheel_base / np.tan(max_steering_angle)
    return DubinsParam(min_radius=rmin)


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # +90 degrees
    perpendicular_direction = np.array(
        [np.cos(current_config.theta - np.pi / 2), np.sin(current_config.theta - np.pi / 2)]
    )

    left_circle = Curve.create_circle(
        center=SE2Transform(current_config.p - radius * perpendicular_direction, 0),
        config_on_circle=current_config,
        radius=radius,
        curve_type=DubinsSegmentType.LEFT,
    )

    right_circle = Curve.create_circle(
        center=SE2Transform(current_config.p + radius * perpendicular_direction, 0),
        config_on_circle=current_config,
        radius=radius,
        curve_type=DubinsSegmentType.RIGHT,
    )
    return TurningCircle(left=left_circle, right=right_circle)


def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> list[Line]:
    # TODO implement here your solution

    if circle_start.type == circle_end.type:
        if np.linalg.norm(circle_start.center.p - circle_end.center.p) <= 1e-3:
            return [
                Line(
                    SE2Transform(circle_start.start_config.p, circle_start.start_config.theta),
                    SE2Transform(circle_start.start_config.p, circle_start.start_config.theta),
                )
            ]
        line_between_centers = Line(circle_start.center, circle_end.center)
        direction = line_between_centers.direction
        perpendicular_direction = np.array([-direction[1], direction[0]])  # +90 degrees
        if circle_start.type == DubinsSegmentType.RIGHT:
            start_pos = circle_start.center.p + circle_start.radius * perpendicular_direction
            end_pos = circle_end.center.p + circle_end.radius * perpendicular_direction
            start_config = SE2Transform(start_pos, 0)
            end_config = SE2Transform(end_pos, 0)
        else:
            start_pos = circle_start.center.p - circle_start.radius * perpendicular_direction
            end_pos = circle_end.center.p - circle_end.radius * perpendicular_direction
            start_config = SE2Transform(start_pos, 0)
            end_config = SE2Transform(end_pos, 0)

        line_between_circles = Line(start_config, end_config)
        direction = line_between_circles.direction
        theta = np.arctan2(direction[1], direction[0])
        start_config.theta = theta
        end_config.theta = theta
        return [Line(start_config, end_config)]
    elif np.linalg.norm(circle_end.center.p - circle_start.center.p) < 2 * circle_start.radius - 1e-3:
        return []
    elif circle_start.type == DubinsSegmentType.RIGHT:
        safe_arg = np.clip(2 * circle_start.radius / np.linalg.norm(circle_end.center.p - circle_start.center.p), -1, 1)
        alpha = np.arccos(safe_arg)  # angle wrt to C line, [0,pi]

        theta = alpha + np.arctan2(
            circle_end.center.p[1] - circle_start.center.p[1], circle_end.center.p[0] - circle_start.center.p[0]
        )

    elif circle_start.type == DubinsSegmentType.LEFT:
        safe_arg = np.clip(2 * circle_start.radius / np.linalg.norm(circle_end.center.p - circle_start.center.p), -1, 1)
        alpha = np.arccos(safe_arg)  # angle wrt to C line, [0,pi]

        theta = -alpha + np.arctan2(
            circle_end.center.p[1] - circle_start.center.p[1], circle_end.center.p[0] - circle_start.center.p[0]
        )

    start_pos = circle_start.center.p + circle_start.radius * np.array([np.cos(theta), np.sin(theta)])
    end_pos = circle_end.center.p - circle_end.radius * np.array([np.cos(theta), np.sin(theta)])
    start_config = SE2Transform(start_pos, 0)
    end_config = SE2Transform(end_pos, 0)

    tangent_line = Line(start_config, end_config)
    if tangent_line.length < 1e-3:
        line_between_centers = Line(circle_start.center, circle_end.center)
        direction = line_between_centers.direction
        if circle_start.type == DubinsSegmentType.RIGHT:
            perpendicular_direction = np.array([direction[1], -direction[0]])  # -90 degrees
            theta = np.arctan2(perpendicular_direction[1], perpendicular_direction[0])
        else:
            perpendicular_direction = np.array([-direction[1], direction[0]])  # +90 degrees
            theta = np.arctan2(perpendicular_direction[1], perpendicular_direction[0])
    else:
        direction = tangent_line.direction
        theta = np.arctan2(direction[1], direction[0])

    start_config.theta = theta
    end_config.theta = theta
    # print("tangente tra:", start_config, end_config)
    return [Line(start_config, end_config)]


def calculate_tangent_circles(circle_start: Curve, circle_end: Curve) -> list[Curve]:
    assert circle_start.type == circle_end.type, "for RLR or LRL circles must be of the same type"

    centers_distance = np.linalg.norm(circle_end.center.p - circle_start.center.p)
    if centers_distance > 4 * circle_start.radius + 1e-3:
        return []
    theta = np.arccos(centers_distance / (4 * circle_start.radius))
    h = 2 * circle_start.radius * np.sin(theta)
    line_between_centers = Line(circle_start.center, circle_end.center)
    direction = line_between_centers.direction
    perpendicular_direction = np.array([-direction[1], direction[0]])  # +90 degrees
    center1 = circle_start.center.p + line_between_centers.length / 2 * direction + h * perpendicular_direction
    center2 = circle_start.center.p + line_between_centers.length / 2 * direction - h * perpendicular_direction

    heading = 0 if circle_start.type == DubinsSegmentType.RIGHT else np.pi

    circle1 = Curve.create_circle(
        center=SE2Transform(center1, 0),
        config_on_circle=SE2Transform(center1 + np.array([circle_start.radius, 0]), heading),
        radius=circle_start.radius,
        curve_type=DubinsSegmentType.RIGHT if circle_start.type == DubinsSegmentType.LEFT else DubinsSegmentType.LEFT,
    )

    circle2 = Curve.create_circle(
        center=SE2Transform(center2, 0),
        config_on_circle=SE2Transform(center2 + np.array([circle_start.radius, 0]), heading),
        radius=circle_start.radius,
        curve_type=DubinsSegmentType.RIGHT if circle_start.type == DubinsSegmentType.LEFT else DubinsSegmentType.LEFT,
    )

    return [circle1, circle2]


def calculate_path_with_straight(start_circle: Curve, end_circle: Curve) -> list[Segment]:
    # used for RSR, LSL, RSL, LSR
    segment = calculate_tangent_btw_circles(start_circle, end_circle)
    if not segment:
        return []
    curve1 = Curve(
        SE2Transform(start_circle.start_config.p, start_circle.start_config.theta),
        SE2Transform(segment[0].start_config.p, segment[0].start_config.theta),
        start_circle.center,
        start_circle.radius,
        start_circle.type,
    )
    set_curve_arc(curve1)

    curve2 = Curve(
        SE2Transform(segment[0].end_config.p, segment[0].end_config.theta),
        SE2Transform(end_circle.end_config.p, end_circle.end_config.theta),
        end_circle.center,
        end_circle.radius,
        end_circle.type,
    )
    set_curve_arc(curve2)

    return [curve1, segment[0], curve2]


def calculate_path_without_straight(start_circle: Curve, end_circle: Curve) -> list[Segment]:
    # used for RLR, LRL
    l = []
    circles = calculate_tangent_circles(start_circle, end_circle)
    for c in circles:
        p1 = calculate_tangent_btw_circles(start_circle, c)
        p2 = calculate_tangent_btw_circles(c, end_circle)
        if p1 and p2:
            curve1 = Curve(
                SE2Transform(start_circle.start_config.p, start_circle.start_config.theta),
                SE2Transform(p1[0].start_config.p, p1[0].start_config.theta),
                start_circle.center,
                start_circle.radius,
                start_circle.type,
            )
            set_curve_arc(curve1)

            curve2 = Curve(
                SE2Transform(p1[0].end_config.p, p1[0].end_config.theta),
                SE2Transform(p2[0].start_config.p, p2[0].start_config.theta),
                c.center,
                c.radius,
                c.type,
            )
            set_curve_arc(curve2)

            curve3 = Curve(
                SE2Transform(p2[0].end_config.p, p2[0].end_config.theta),
                SE2Transform(end_circle.end_config.p, end_circle.end_config.theta),
                end_circle.center,
                end_circle.radius,
                end_circle.type,
            )
            set_curve_arc(curve3)
            l.append([curve1, curve2, curve3])
    return l


def set_curve_arc(curve: Curve):
    if curve.type == DubinsSegmentType.RIGHT:
        curve.arc_angle = curve.start_config.theta - curve.end_config.theta
    else:
        curve.arc_angle = curve.end_config.theta - curve.start_config.theta


def print_path(path: list[Segment]):
    for p in path:
        if isinstance(p, Curve):
            print(p, p.start_config, p.end_config, p.center, p.radius)
        else:
            print(p, p.start_config, p.end_config)
    print("\n")


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!

    # start circles
    turning_circles = calculate_turning_circles(start_config, radius)
    left_circle_start = turning_circles.left
    right_circle_start = turning_circles.right

    # end circles
    turning_circles = calculate_turning_circles(end_config, radius)
    left_circle_end = turning_circles.left
    right_circle_end = turning_circles.right

    # print(left_circle_start.type)

    paths = []

    # path RSR
    paths.append(calculate_path_with_straight(right_circle_start, right_circle_end))

    # path LSL
    paths.append(calculate_path_with_straight(left_circle_start, left_circle_end))

    # path RSL
    paths.append(calculate_path_with_straight(right_circle_start, left_circle_end))

    # path LSR
    paths.append(calculate_path_with_straight(left_circle_start, right_circle_end))

    # path RLR
    for p in calculate_path_without_straight(right_circle_start, right_circle_end):
        paths.append(p)

    # path LRL
    for p in calculate_path_without_straight(left_circle_start, left_circle_end):
        paths.append(p)

    # find the shortest path
    min_length = np.inf
    shortest_path = []
    for p in paths:
        path_length = 0

        for s in p:
            length = s.length
            path_length += length
        # print(p)
        if path_length < min_length and path_length > 0:
            min_length = path_length
            shortest_path = p
    # print_path(shortest_path)
    return shortest_path  # e.g., [Curve(), Line(),..]


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
    shortest_forward_path = calculate_dubins_path(start_config, end_config, radius)

    # reverse the path
    start_config_reverse = SE2Transform(start_config.p, start_config.theta + np.pi)
    end_config_reverse = SE2Transform(end_config.p, end_config.theta + np.pi)
    shortest_reverse_path = calculate_dubins_path(start_config_reverse, end_config_reverse, radius)

    reverse_path_length = 0
    forward_path_length = 0

    for s in shortest_reverse_path:
        s.gear = Gear.REVERSE
        s.start_config.theta -= np.pi
        s.end_config.theta -= np.pi
        if s.type == DubinsSegmentType.RIGHT:
            s.type = DubinsSegmentType.LEFT
        elif s.type == DubinsSegmentType.LEFT:
            s.type = DubinsSegmentType.RIGHT
        reverse_path_length += s.length

    for s in shortest_forward_path:
        forward_path_length += s.length

    if forward_path_length < reverse_path_length:
        p = shortest_forward_path
    else:
        p = shortest_reverse_path
    print_path(p)
    return p
