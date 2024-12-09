from mimetypes import init
import random
from dataclasses import dataclass
from turtle import left
from typing import Sequence, Callable, Tuple, Optional

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.controllers.pure_pursuit import PurePursuit
from dg_commons.controllers.speed import SpeedController
from dg_commons.controllers.steer import SteerController
from dg_commons.maps.lanes import DgLanelet

from geometry import SE2_from_xytheta

from networkx import center
import numpy as np
import sympy as spy


class VehicleDyn:
    vg: VehicleGeometry
    vp: VehicleParameters

    x: spy.Matrix
    u: spy.Matrix

    n_x: int
    n_u: int

    f: spy.Function
    A: spy.Function
    B: spy.Function

    def __init__(self, vg: VehicleGeometry, vp: VehicleParameters):
        self.vg = vg
        self.vp = vp

        self.x = spy.Matrix(spy.symbols("x y psi vx delta", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust ddelta", real=True))  # inputs

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        0x 1y 2psi 3vx 6delta
        """
        # Dynamics
        f = spy.zeros(self.n_x, 1)
        x, y, psi, vx, delta = self.x
        acc, ddelta = self.u

        dtheta = vx * spy.tan(delta) / self.vg.wheelbase
        vy = dtheta * self.vg.lr
        costh = spy.cos(psi)
        sinth = spy.sin(psi)
        xdot = vx * costh - vy * sinth
        ydot = vx * sinth + vy * costh
        # Define the dynamics equations
        f[0] = xdot  # dx/dt
        f[1] = ydot  # dy/dt
        f[2] = dtheta  # dpsi/dt
        f[3] = acc  # dvx/dt
        f[4] = ddelta  # ddelta/dt

        A = f.jacobian(self.x)
        B = f.jacobian(self.u)

        f_func = spy.lambdify((self.x, self.u), f, "numpy")
        A_func = spy.lambdify((self.x, self.u), A, "numpy")
        B_func = spy.lambdify((self.x, self.u), B, "numpy")

        return f_func, A_func, B_func

    def get_discrete_time_dynamics_approx(self, dt: float) -> tuple[Callable, Callable, Callable]:
        f, A, B = self.get_dynamics()

        def A_d_single(x, u):
            return np.eye(self.n_x) + A(x, u) * dt

        def B_d_single(x, u):
            return B(x, u) * dt

        def f_d_single(x, u):
            return x + f(x, u) * dt

        def A_d(x_traj, u_traj):
            """Vectorized version of A_d_single, takes (n_x, N) and (n_u, N) and returns (N,n_x,n_x)"""
            if len(x_traj.shape) == 1:
                x_traj = x_traj.reshape(-1, 1)
                u_traj = u_traj.reshape(-1, 1)
            vectorized_A = np.vectorize(A_d_single, signature="(n_x), (n_u) -> (n_x,n_x)")
            return vectorized_A(x_traj.T, u_traj.T)

        def B_d(x_traj, u_traj):
            """Vectorized version of B_d_single, takes (n_x, N) and (n_u, N) and returns (N,n_x,n_u)"""
            if len(x_traj.shape) == 1:
                x_traj = x_traj.reshape(-1, 1)
                u_traj = u_traj.reshape(-1, 1)
            vectorized_B = np.vectorize(B_d_single, signature="(n_x), (n_u)-> (n_x,n_u)")
            return vectorized_B(x_traj.T, u_traj.T)

        def f_d(x_traj, u_traj):
            """Vectorized version of f_d_single, takes (n_x, N) and (n_u, N) and returns (N,n_x)"""
            if len(x_traj.shape) == 1:
                x_traj = x_traj.reshape(-1, 1)
                u_traj = u_traj.reshape(-1, 1)
            vectorized_f = np.vectorize(f_d_single, signature="(n_x), (n_u) -> (n_x)")
            return vectorized_f(x_traj.T, u_traj.T)

        return f_d, A_d, B_d


class PurePursuitController:
    def __init__(self, path: Lanelet):
        self.speed_controller = HL_SpeedController()
        self.steer_controller = HL_SteerController(path)

    def compute_control(self, current_state: VehicleState, current_time: float) -> VehicleCommands:
        ddelta = self.steer_controller.get_control(current_state, current_time)
        acc = self.speed_controller.get_control(current_state, current_time)

        return VehicleCommands(acc=acc, ddelta=ddelta)

    def update_path(self, new_path: Lanelet):
        self.steer_controller.update_reference_path(new_path)

    def update_speed_reference(self, reference: Optional[float] = None):
        self.speed_controller.controller.update_reference(reference)


class HL_SpeedController:
    def __init__(self):
        self.controller = SpeedController()
        self.update_speed_reference()

    def get_control(self, current_state: VehicleState, current_time: float) -> float:
        """Call low level controller to get control"""
        self.controller.update_measurement(current_state.vx)
        return self.controller.get_control(at=current_time)

    def update_speed_reference(self, reference: Optional[float] = None):
        """TODO, ideally should be an intelligent speed reference considering obstacles and stuff"""
        if reference is None:  # let the controller decide (stupid logic for now)
            constant_speed = 10
            reference = constant_speed
            self.controller.update_reference(constant_speed)
        else:  # set the reference to the given value
            self.controller.update_reference(reference)


class HL_SteerController:
    def __init__(self, path: Lanelet):
        self.LL_controller = SteerController()
        self.steer_controller = PurePursuit()
        self.steer_controller.update_path(DgLanelet.from_commonroad_lanelet(path))
        self.path = path

    def get_control(self, current_state: VehicleState, current_time: float) -> float:
        """Call low level controller to get control"""
        self.LL_controller.update_measurement(current_state.delta)

        # purepuruit needs to know how far the vehicle is along the path
        position_diff_square = np.sum(
            (self.path.center_vertices - np.array([current_state.x, current_state.y])) ** 2, axis=1
        )
        closest_vertex_index = np.argmin(position_diff_square)
        distance_along_path = self.path.distance[closest_vertex_index] + np.sqrt(
            position_diff_square[closest_vertex_index]
        )

        # update pose and speed
        self.steer_controller.update_pose(
            SE2_from_xytheta((current_state.x, current_state.y, current_state.psi)), along_path=distance_along_path
        )
        self.steer_controller.update_speed(current_state.vx)

        # get desired steering, update low level reference (delta) and get control (ddelta)
        reference = self.steer_controller.get_desired_steering()
        self.LL_controller.update_reference(reference)

        return self.LL_controller.get_control(current_time)

    def update_reference_path(self, new_path: Lanelet):
        self.steer_controller.update_path(DgLanelet.from_commonroad_lanelet(new_path))


@dataclass
class MPCController:
    # planner: Planner
    horizon: int
    dt: float

    def compute_control(self, current_state: np.ndarray, goal: PlanningGoal) -> VehicleCommands:
        # Implement the MPC control logic here
        pass


if __name__ == "__main__":
    # Define the vehicle geometry and parameters
    vg = VehicleGeometry.default_bicycle()
    vp = VehicleParameters.default_bicycle()

    # Define the dynamics
    dyn = VehicleDyn(vg=vg, vp=vp)
    f, A, B = dyn.get_dynamics()
    f_d, A_d, B_d = dyn.get_discrete_time_dynamics_approx(dt=0.1)
    x_traj = np.ones([5, 10])
    u_traj = np.ones([2, 10])

    # define the controller
    left_vertices = np.array([[0, 1], [1, 1], [2, 1], [3, 1]])
    center_vertices = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    right_vertices = np.array([[0, -1], [1, -1], [2, -1], [3, -1]])
    path = Lanelet(left_vertices, center_vertices, right_vertices, 0)
    controller = PurePursuitController(path)
    state = np.zeros(5)
    controls = VehicleCommands(0, 0)
    for i in range(10):
        A = A_d(state, controls.as_ndarray())[0]
        B = B_d(state, controls.as_ndarray())[0]
        # print(f"A: {A}, B: {B}")
        controls = controller.compute_control(VehicleState.from_array(state), i)
        print(controls)
        state = A @ state + B @ controls.as_ndarray()
        print(state)
