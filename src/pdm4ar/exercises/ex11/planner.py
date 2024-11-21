from calendar import c
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

    old_objective: float
    new_objective: float

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
        self.n_x = self.spaceship.n_x
        self.n_u = self.spaceship.n_u
        self.n_p = self.spaceship.n_p

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()
        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # TODO DEBUG (FAKE GOAL, HAS TO BE DELETED)
        self.goal_state = DynObstacleState(10, 10, 0.1, 0.1, 0.1, 0)
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
            self.old_objective = objective.value
            self.problem = cvx.Problem(objective, self._get_constraints())
            try:
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
                print(f"Iteration {iteration + 1} error: {error}")
                print(f"Iteration {iteration + 1} objective value: {objective.value}")
                self.new_objective = objective.value
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

            if self._check_convergence():
                break
        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array(
            tuple(range(self.params.K)), self.U_bar[0], self.U_bar[1], self.X_bar
        )

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
        # self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K), name="X"),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K), name="U"),
            "p": cvx.Variable(self.spaceship.n_p, name="p", integer=True),
            "v_dyn": cvx.Variable((self.spaceship.n_x, self.params.K - 1), name="v_dyn"),
            # "delta": cvx.Variable(nonneg=True, name="delta"),
        }
        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x, name="init_state"),
            "goal_config": cvx.Parameter(6, name="goal_config"),
            # TODO
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        x_goal = self.problem_parameters["goal_config"][0]
        y_goal = self.problem_parameters["goal_config"][1]
        # goal_coords = cvx.Parameter(2)
        goal_coords = cvx.vstack([x_goal, y_goal])

        x_fin = self.variables["X"][0, self.params.K - 1]
        y_fin = self.variables["X"][1, self.params.K - 1]
        # fin_coords = cvx.Parameter(2)
        fin_coords = cvx.vstack([x_fin, y_fin])

        pose_goal = self.problem_parameters["goal_config"][2]
        pose_fin = self.variables["X"][2, self.params.K - 1]
        # Cambiato visto che l'operazione modulo (%) non si può usare in cvxpy
        # parte_intera = cvx.floor(pose_fin - pose_goal + np.pi / 2 * np.pi)
        # pose_diff = pose_fin - pose_goal + np.pi - 2 * np.pi * parte_intera - np.pi
        # Distanza diretta e complementare

        # delta_1 = cvx.norm(pose_fin - pose_goal, 1)
        # delta_2 = 2 * np.pi - delta_1

        # Vincoli
        constraints = []
        vx_fin = self.variables["X"][3, self.params.K - 1]
        vy_fin = self.variables["X"][4, self.params.K - 1]
        v_fin = cvx.vstack([vx_fin, vy_fin])

        vx_goal = self.problem_parameters["goal_config"][3]
        vy_goal = self.problem_parameters["goal_config"][4]
        v_goal = cvx.vstack([vx_goal, vy_goal])

        A, B_plus, B_minus, F, r = self._convexification()
        X_k_plus_1 = self.variables["X"][:, 1 : self.params.K]
        print(f"Dimensions of X_k_plus_1: {X_k_plus_1.shape}")
        print("A ", A.shape)
        print("X_K ", self.variables["X"][:, 0 : self.params.K - 1].shape)
        print("B_plus ", B_plus.shape)
        print("U_PLUS ", self.variables["U"][:, 1 : self.params.K].shape)
        print("B_minus ", B_minus.shape)
        print("U_minus ", self.variables["U"][:, 0 : self.params.K - 1].shape)
        print("F ", F.shape)
        print("p ", self.variables["p"].shape)
        print("r ", r.shape)
        print("v_dyn ", self.variables["v_dyn"].shape)

        # print("Ak * xk", np.matmul(A, self.variables["X"][:, 0 : self.params.K - 1].T).shape)
        # print("B_plus * uk+1", np.matmul(B_plus, self.variables["U"][:, 1 : self.params.K].T).shape)
        # print("B_minus * uk", np.matmul(B_minus, self.variables["U"][:, 0 : self.params.K - 1].T).shape)
        # print("F * p", np.matmul(F, self.variables["p"].T).shape)

        constraints = [
            # Initial state costraint (WAS ALREADY IN THE EXAMPLE)
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],
            # Input constraints
            self.variables["U"][:, 0] == np.zeros(self.spaceship.n_u),
            self.variables["U"][:, self.params.K - 1] == np.zeros(self.spaceship.n_u),
            # Needs to be close to the goal
            cvx.norm(goal_coords - fin_coords) <= self.params.stop_crit,
            # Orientation constraint
            # delta_1 >= self.variables["delta"],
            # delta_2 >= self.variables["delta"],
            # self.variables["delta"] <= self.params.stop_crit,
            pose_fin - pose_goal <= self.params.stop_crit,
            # Specified velocity constraint
            cvx.norm(v_fin - v_goal) <= self.params.stop_crit,
            # No collisions
            # TODO
            # Mass constraint
            self.variables["X"][7, :] >= self.sg.m,
            # Thrust constraint
            self.variables["U"][0, :] >= self.sp.thrust_limits[0],
            self.variables["U"][0] <= self.sp.thrust_limits[1],
            # Thruster angle costraint
            self.variables["X"][6, :] >= self.sp.delta_limits[0],
            self.variables["X"][6, :] <= self.sp.delta_limits[1],
            # Maximum time
            # TODO
            # Rate of change
            self.variables["U"][1, :] >= self.sp.ddelta_limits[0],
            self.variables["U"][1] <= self.sp.ddelta_limits[1],
            # Missing dynamics constraints
            # Should we add 39c,39d,39e?
            # Are 39f, 39d already in here?
        ]

        # Dynamic constraints
        # TODO vettorizzare usando cvx.matmul se si riesce, ma si lamenta perchè non sa gestire 3 dimensioni
        constraints += [
            self.variables["X"][:, i]
            == cvx.matmul(A[i - 1], self.variables["X"][:, i - 1].T)
            + cvx.matmul(B_plus[i - 1], self.variables["U"][:, i].T)
            + cvx.matmul(B_minus[i - 1], self.variables["U"][:, i - 1].T)
            + cvx.matmul(F[i - 1], self.variables["p"].T)
            + r[i - 1]
            + self.variables["v_dyn"][:, i - 1]
            for i in range(1, self.params.K)
        ]

        constraints += [
            cvx.norm(self.variables["X"][:6, i] - self.problem_parameters["goal_config"]) <= self.params.stop_crit
            for i in range(self.variables["p"].value, self.params.K)
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
        self.problem_parameters["goal_config"].value = self.goal_state.as_ndarray()
        # TODO populate other problem parameters + function is not modifying anything
        return (
            A_bar.reshape((-1, self.n_x, self.n_x), order="F"),
            B_plus_bar.reshape((-1, self.n_x, self.n_u), order="F"),
            B_minus_bar.reshape((-1, self.n_x, self.n_u), order="F"),
            F_bar.reshape((-1, self.n_x, self.n_p), order="F"),
            r_bar.reshape((-1, self.n_x), order="F"),
        )

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        return abs(self.new_objective - self.old_objective) < self.params.stop_crit * abs(self.old_objective)

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
    goal_state = DynObstacleState(10, 10, 0.1, 0.1, 0.1, 0)
    mycmds, mystates = planner.compute_trajectory(init_state, goal_state)
