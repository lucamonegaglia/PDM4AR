from dataclasses import dataclass, field
from hmac import new
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

    def compute_trajectory(  # MISSING TRUST REGION UPDATE, VIRUTAL CONTROL???
        self, init_state: SpaceshipState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state

        # Initialize state and control trajectories
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        max_iterations = self.params.max_iterations
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}")
            self._convexification()
            objective = self._get_objective()
            old_objective = objective.value
            self.problem = cvx.Problem(objective, self._get_constraints())
            try:  # QUESTION: When and how should I use error?
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
                print(f"Iteration {iteration + 1} error: {error}")
                print(f"Iteration {iteration + 1} objective value: {objective.value}")
                new_objective = objective.value
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            # Check the solution status
            if self.problem.status == cvx.OPTIMAL:
                print("Optimal solution found.")
            elif self.problem.status == cvx.INFEASIBLE:
                print("Problem is infeasible.")
            #    return None, None
            elif self.problem.status == cvx.UNBOUNDED:
                print("Problem is unbounded.")
            #    return None, None
            else:
                print("Solver did not converge.")
            #    return None, None

            # Update the trajectories with the new solution
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value

            if self._check_convergence(new_objective, old_objective):
                break
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
            "v_dyn": cvx.Variable((self.spaceship.n_x, self.params.K - 1)),
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
        pose_diff = (pose_fin - pose_goal + np.pi) % (2 * np.pi) - np.pi

        vx_final = self.variables["X"][3][self.params.K - 1]
        vy_final = self.variables["X"][4][self.params.K - 1]
        vx_goal = self.goal_state.idx[3]
        vy_goal = self.goal_state.idx[4]
        #

        A, B_plus, B_minus, F, r = self._convexification()

        constraints = [
            # Initial state costraint (WAS ALREADY IN THE EXAMPLE)
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],
            # Input constraints
            self.variables["U"][:][0] == np.zeros(self.spaceship.n_u),
            self.variables["U"][:][self.params.K - 1] == np.zeros(self.spaceship.n_u),
            # Needs to be close to the goal
            np.linalg.norm([x_fin - x_goal, y_fin - y_goal]) <= self.params.stop_crit,
            # Orientation constraint
            np.linalg.norm(pose_diff) <= self.params.stop_crit,
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
            # Dynamics constraints
            self.variables["X"][:, 1 : self.params.K]
            == A @ self.variables["X"][:, 0 : self.params.K - 1]
            + B_plus @ self.variables["U"][:, 1 : self.params.K]
            + B_minus @ self.variables["U"][:, 0 : self.params.K - 1]
            + F @ self.variables["p"]
            + r
            + self.variables["v_dyn"],
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
        return A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar

    def _check_convergence(self, new_objective, old_objective) -> bool:
        """
        Check convergence of SCvx.
        """
        return np.abs(new_objective - old_objective) < self.params.stop_crit * abs(old_objective)

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        # TODO
        pass

    @staticmethod
    def _extract_seq_from_array(
        ts: tuple[int, ...], F: np.ndarray, ddelta: np.ndarray, npstates: np.ndarray
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Create DgSampledSequence from provided numpy arrays and timestamps.

        Parameters:
            ts (tuple[int, ...]): Timestamps corresponding to the samples.
            F (np.ndarray): Force or action inputs (1D array).
            ddelta (np.ndarray): Secondary inputs, e.g., deltas or accelerations (1D array).
            npstates (np.ndarray): State trajectory (2D array with one row per timestamp).

        Returns:
            tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
            Command and state sequences.
        """
        # Validate inputs
        assert (
            len(ts) == F.shape[0] == ddelta.shape[0] == npstates.shape[0]
        ), "All inputs must have the same length as the number of timestamps."

        # Create the command sequence
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # Create the state sequence
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)

        return mycmds, mystates


# incompleto, non dovrebbe servire.
if __name__ == "__main__":
    sg = SpaceshipGeometry(
        color="royalblue",
        m=2.0,  # MASS TO BE INTENDED AS MASS OF THE ROCKET WITHOUT FUEL
        Iz=1e-00,
        w_half=0.4,
        l_c=0.8,
        l_f=0.5,
        l_r=1,
        l_t_half=0.3,
        e=0.1,  # Example value for 'e'
        w_t_half=0.3,
        F_max=3.0,
    )  # Example value for 'F_max'

    sp = SpaceshipParameters(
        m_v=2.0,
        C_T=0.01,
        vx_limits=(-10 / 3.6, 10 / 3.6),
        acc_limits=(-1.0, 1.0),
        thrust_limits=(-2.0, 2.0),
        delta_limits=(-np.deg2rad(60), np.deg2rad(60)),
        ddelta_limits=(-np.deg2rad(45), np.deg2rad(45)),
    )
    planets = {}
    satellites = {}
    planner = SpaceshipPlanner(planets, satellites, sg, sp)
    init_state = SpaceshipState(0, 0, 0, 0, 0, 0, 0, 2)
    goal_state = DynObstacleState(10, 10, 0, 0.1, 0.1, 0)
    mycmds, mystates = planner.compute_trajectory(init_state, goal_state)
