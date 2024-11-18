from dataclasses import dataclass, field
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)

from pdm4ar.exercises.ex11.discretization import *
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SpaceshipState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state

        #
        # TODO: Implement SCvx algorithm or comparable
        #

        self._convexification()
        try:
            error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
        except cvx.SolverError:
            print(f"SolverError: {self.params.solver} failed to solve the problem.")

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array()

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        # TODO
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        U = np.zeros((self.spaceship.n_u, K))
        p = np.zeros((self.spaceship.n_p))

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        # TODO what is (6,1)?
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K)),
            "p": cvx.Variable(self.spaceship.n_p),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x)
            # TODO
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        x_goal = self.goal_state.idx[0]
        y_goal = self.goal_state.idx[1]
        x_fin = self.variables["X"][0][self.params.K - 1]
        y_fin = self.variables["X"][1][self.params.K - 1]

        pose_goal = self.goal_state.idx[2]
        pose_fin = self.variables["X"][2][self.params.K - 1]

        vx_final = self.variables["X"][3][self.params.K - 1]
        vy_final = self.variables["X"][4][self.params.K - 1]
        vx_goal = self.goal_state.idx[3]
        vy_goal = self.goal_state.idx[4]
        #
        constraints = [
            # Initial state costraint (WAS ALREADY IN THE EXAMPLE)
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],
            # Input constraints
            self.variables["U"][:][0] == np.zeros(self.spaceship.n_u),
            self.variables["U"][:][self.params.K - 1] == np.zeros(self.spaceship.n_u),
            # Needs to be close to the goal
            np.linalg.norm([x_fin - x_goal, y_fin - y_goal]) <= self.params.stop_crit,
            # Orientation constraint
            np.linalg.norm(pose_fin - pose_goal) <= self.params.stop_crit,
            # Specified velocity constraint
            np.linalg.norm([vx_final - vx_goal, vy_final - vy_goal]) <= self.params.stop_crit,
            # No collisions
            # TODO
            # Mass constraint
            self.variables["X"][7] >= self.sg.m,
            # Thrust constraint
            self.variables["U"][0] >= self.sp.thrust_limits[0] and self.variables["U"][0] <= self.sp.thrust_limits[1],
            # Thruster angle costraint
            self.variables["X"][6][:] >= self.sp.delta_limits[0]
            and self.variables["X"][6][:] <= self.sp.delta_limits[1],
            # Maximum time
            # TODO
            # Rate of change
            self.variables["U"][1] >= self.sp.ddelta_limits[0] and self.variables["U"][1] <= self.sp.ddelta_limits[1],
            # Missing dynamics constraints
            # Should we add 39c,39d,39e?
            # Are 39f, 39d already in here?
        ]
        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # TODO
        # Example objective
        objective = self.params.weight_p @ self.variables["p"]

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        # TODO populate other problem parameters + function is not modifying anything

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        # TODO
        pass

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        # TODO
        pass

    @staticmethod
    def _extract_seq_from_array() -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        # TODO repurpose this to take our arrays as parameters?
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F = np.array([0, 1, 2, 3, 4])
        ddelta = np.array([0, 0, 0, 0, 0])
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 8)
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)
        return mycmds, mystates
