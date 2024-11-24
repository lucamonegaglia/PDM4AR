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

from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Set the backend to 'Agg' for non-GUI use


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = True  # if True, the optimization steps are shown
    max_iterations: int = 20  # max algorithm iterations

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
        init_state: SpaceshipState,
        goal_state: DynObstacleState,
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

        # TODO DEBUG (Temporary goal and init state, used in convex)
        # self.goal_state = DynObstacleState(0, 0, 0, 0, 0, 0)
        # self.init_state = SpaceshipState(0, 0, 0, 0, 0, 0, 0, 2)

        self.init_state = init_state
        self.goal_state = goal_state

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()
        # assign trust region radius
        self.problem_parameters["radius_trust_region"].value = self.params.tr_radius

        self.initial_guess()  # update directly the problem parameters

        # Variables
        self.variables = self._get_variables()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective2()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(  # MISSING TRUST REGION UPDATE, VIRUTAL CONTROL???
        self, init_state: SpaceshipState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        # overwrite previous (temporary) init and goal state
        self.init_state = init_state
        self.goal_state = goal_state

        # Initialize state and control trajectories
        self.initial_guess()

        # used for plotting only
        self.v_dyn = np.zeros((self.spaceship.n_x, self.params.K - 1))

        max_iterations = self.params.max_iterations

        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}")
            self._convexification()
            # plot before solving
            self.plot_predicted_and_real_results(iteration)

            if iteration != 0:
                self.old_objective = self.problem.objective.value
            try:
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
                print(f"Iteration {iteration + 1} error: {error}")
                print(f"Iteration {iteration + 1} objective value: {self.problem.objective.value}")
                self.new_objective = self.problem.objective.value
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
            # Check for constraint violations
            # for constraint in self.problem.constraints:
            #    if constraint.dual_value is not None and np.any(constraint.dual_value < 0):
            #        print(f"Violated Constraint: {constraint}, Value: {constraint.dual_value}")
            # Check the solution status
            if self.problem.status == cvx.OPTIMAL:
                print("Optimal solution found.")
            elif self.problem.status == cvx.INFEASIBLE:
                print("Problem is infeasible. Check constraints or feasibility.")
            elif self.problem.status == cvx.UNBOUNDED:
                print("Problem is unbounded. Check if constraints are missing.")
            elif self.problem.status == cvx.OPTIMAL_INACCURATE:
                print(
                    "Solution is near-optimal but not accurate. Reason: Numerical instability, insufficient solver iterations, or precision issues.."
                )
            elif self.problem.status == cvx.INFEASIBLE_INACCURATE:
                print(
                    "Problem may be infeasible, but the solver could not confirm. Poor scaling, tight constraints, or solver limitations."
                )
            elif self.problem.status == cvx.UNBOUNDED_INACCURATE:
                print("Problem may be unbounded, but the solver could not confirm. Add bounds or regularization.")
            elif self.problem.status == cvx.USER_LIMIT:
                print("Solver stopped due to user-defined iteration or time limits. Increase the limits.")
            else:
                print("Solver did not converge. Check formulation and solver settings.")
            temp_X = self.variables["X"].value.copy()
            temp_U = self.variables["U"].value.copy()
            if iteration != 0 and self._check_convergence2(temp_X, temp_U):
                print("Convergence reached.")
                break
            # compute flow_map
            flow_map_pre_solver = self.integrator.integrate_nonlinear_piecewise(
                self.problem_parameters["X_bar"].value,
                self.problem_parameters["U_bar"].value,
                self.problem_parameters["p_bar"].value,
            )
            delta_pre_solver = flow_map_pre_solver - self.problem_parameters["X_bar"].value
            objective_pre_solver_non_discretized = self._get_non_discretize_objective2(delta_pre_solver)

            flow_map_post_solver = self.integrator.integrate_nonlinear_piecewise(
                self.variables["X"].value, self.variables["U"].value, self.variables["p"].value
            )
            delta_post_solver = flow_map_post_solver - self.variables["X"].value
            objective_post_solver_non_discretized = self._get_non_discretize_objective2(delta_post_solver)

            rho = (objective_pre_solver_non_discretized - objective_post_solver_non_discretized) / (
                objective_pre_solver_non_discretized - self.new_objective
            )

            if rho >= self.params.rho_0:
                self.problem_parameters["X_bar"].value = self.variables["X"].value
                self.problem_parameters["U_bar"].value = self.variables["U"].value
                self.problem_parameters["p_bar"].value = self.variables["p"].value

            # update trust region
            self._update_trust_region(rho)

            # used for debug only
            self.v_dyn = self.variables["v_dyn"].value

            # if iteration != 0 and self._check_convergence(objective_pre_solver_non_discretized):
            #    break

        # last plot
        self.plot_predicted_and_real_results("final")

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array(
            # tuple(range(self.params.K)), self.U_bar[0], self.U_bar[1], self.X_bar.T
            tuple(range(self.params.K)),
            self.problem_parameters["U_bar"].value[0],
            self.problem_parameters["U_bar"].value[1],
            self.problem_parameters["X_bar"].T,
        )

        return mycmds, mystates

    def plot_predicted_and_real_results(self, iteration):
        x0 = self.problem_parameters["X_bar"].value[:, 0]
        integrated_X = self.integrator.integrate_nonlinear_full(
            x0, self.problem_parameters["U_bar"].value, self.problem_parameters["p_bar"].value
        )
        fig, axs = plt.subplots(4, 2, figsize=(10, 15))

        # Plot predicted position trajectory
        axs[0][0].plot(self.problem_parameters["X_bar"].value[0, :].T, self.problem_parameters["X_bar"].value[1, :].T)
        axs[0][0].set_title("Predicted Position")
        axs[0][0].set_xlabel("Step")
        axs[0][0].set_ylabel("Position")

        # Plot real position trajectory
        axs[0][1].plot(integrated_X[0, :].T, integrated_X[1, :].T)
        axs[0][1].set_title("Real Position")
        axs[0][1].set_xlabel("Step")
        axs[0][1].set_ylabel("Position")

        # Plot phi trajectory
        axs[1][0].plot(self.problem_parameters["X_bar"].value[2, :].T)
        axs[1][0].set_title("Predicted phi")
        axs[1][0].set_xlabel("Step")
        axs[1][0].set_ylabel("phi ")

        # Plot real phi trajectory
        axs[1][1].plot(integrated_X[2, :].T)
        axs[1][1].set_title("Real phi")
        axs[1][1].set_xlabel("Step")
        axs[1][1].set_ylabel("phi ")

        # Plot predicted velocities
        axs[2][0].plot(self.problem_parameters["X_bar"].value[3, :].T, self.problem_parameters["X_bar"].value[4, :].T)
        axs[2][0].set_title("Predicted Velocities")
        axs[2][0].set_xlabel("Step")
        axs[2][0].set_ylabel("Predicted Velocities")
        axs[2][0].legend(["VX", "VY"])

        # Plot real velocities
        axs[2][1].plot(integrated_X[3, :].T, integrated_X[4, :].T)
        axs[2][1].set_title("Real Velocities")
        axs[2][1].set_xlabel("Step")
        axs[2][1].set_ylabel("Real Velocities")
        axs[2][1].legend(["VX", "VY"])

        # Plot U
        axs[3][0].plot(self.problem_parameters["U_bar"].value.T)
        axs[3][0].set_title("U")
        axs[3][0].set_xlabel("Step")
        axs[3][0].set_ylabel("U")

        # Plot vdyn
        axs[3][1].plot(self.v_dyn)
        axs[3][1].set_title("VDYN")
        axs[3][1].set_xlabel("Step")
        axs[3][1].set_ylabel("VDYN")
        axs[3][1].legend(["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"])

        plt.tight_layout()
        plt.savefig(f"src/pdm4ar/exercises/ex11/plots/plot{iteration}.png")
        plt.close()

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        returns X_bar, U_bar, p_bar initial guess for SCvx.
        """
        # TODO
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        # X[0, :] = self.problem_parameters["init_state"][0]
        # X[1, :] = self.problem_parameters["init_state"][1]
        # X[7, :] = self.problem_parameters["init_state"][7]
        X[7, :] = self.sg.m

        U = np.zeros((self.spaceship.n_u, K))
        p = np.ones((self.spaceship.n_p))

        # update problem parameters
        self.problem_parameters["X_bar"].value = X
        self.problem_parameters["U_bar"].value = U
        self.problem_parameters["p_bar"].value = p

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
            "U": cvx.Variable((self.spaceship.n_u, self.params.K), name="U"),  # 0: thrust, 1: ddelta
            "p": cvx.Variable(self.spaceship.n_p, name="p"),  # final time
            "v_dyn": cvx.Variable((self.spaceship.n_x, self.params.K - 1), name="v_dyn"),
            # "delta": cvx.Variable(nonneg=True, name="delta"),
            "v_init_state": cvx.Variable(self.spaceship.n_x, name="v_init_state"),
            "v_goal_coords": cvx.Variable(name="v_goal_coords"),
            "v_goal_pose": cvx.Variable(name="v_goal_pose"),
            "v_goal_vel": cvx.Variable(name="v_goal_vel"),
        }
        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x, name="init_state"),
            "goal_config": cvx.Parameter(6, name="goal_config"),
            "A_bar": cvx.Parameter((self.n_x * self.n_x, self.params.K - 1), name="A_bar"),  # shape 64 x 49
            "B_plus_bar": cvx.Parameter((self.n_x * self.n_u, self.params.K - 1), name="B_plus_bar"),  # shape 16 x 49
            "B_minus_bar": cvx.Parameter((self.n_x * self.n_u, self.params.K - 1), name="B_minus_bar"),  # shape 16 x 49
            "F_bar": cvx.Parameter((self.n_x * self.n_p, self.params.K - 1), name="F_bar"),  # shape 8 x 49
            "r_bar": cvx.Parameter((self.n_x, self.params.K - 1), name="r_bar"),  # shape 8 x 49
            "radius_trust_region": cvx.Parameter(name="radius_trust_region"),
            "X_bar": cvx.Parameter((self.n_x, self.params.K), name="X_bar"),
            "U_bar": cvx.Parameter((self.n_u, self.params.K), name="U_bar"),
            "p_bar": cvx.Parameter(self.n_p, name="p_bar"),
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

        self._convexification()
        A = self.problem_parameters["A_bar"].T
        B_plus = self.problem_parameters["B_plus_bar"].T
        B_minus = self.problem_parameters["B_minus_bar"].T
        F = self.problem_parameters["F_bar"].T
        r = self.problem_parameters["r_bar"].T
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

        deltax = self.variables["X"] - self.problem_parameters["X_bar"]
        deltau = self.variables["U"] - self.problem_parameters["U_bar"]
        deltap = self.variables["p"] - self.problem_parameters["p_bar"]

        constraints = [
            # Initial state costraint (WAS ALREADY IN THE EXAMPLE)
            self.variables["X"][:, 0] - self.problem_parameters["init_state"] - self.variables["v_init_state"] == 0,
            # Input constraints
            self.variables["U"][:, 0] - np.zeros(self.spaceship.n_u) == 0,
            self.variables["U"][:, self.params.K - 1] - np.zeros(self.spaceship.n_u) == 0,
            # Needs to be close to the goal
            cvx.norm(goal_coords - fin_coords, "fro") - self.variables["v_goal_coords"] <= 0,
            # Orientation constraint
            # delta_1 >= self.variables["delta"],
            # delta_2 >= self.variables["delta"],
            # self.variables["delta"] <= self.params.stop_crit,
            pose_fin - pose_goal - self.variables["v_goal_pose"] == 0,
            # Specified velocity constraint
            cvx.norm(v_fin - v_goal, "fro") - self.variables["v_goal_vel"] <= 0,
            # No collisions
            # TODO
            # Mass constraint
            self.variables["X"][7, :] - self.sg.m >= 0,
            # Thrust constraint
            self.variables["U"][0, :] - self.sp.thrust_limits[0] >= 0,
            self.variables["U"][0, :] - self.sp.thrust_limits[1] <= 0,
            # Thruster angle costraint
            self.variables["X"][6, :] - self.sp.delta_limits[0] >= 0,
            self.variables["X"][6, :] - self.sp.delta_limits[1] <= 0,
            # Maximum time
            # TODO
            # Rate of change
            self.variables["U"][1, :] - self.sp.ddelta_limits[0] >= 0,
            self.variables["U"][1, :] - self.sp.ddelta_limits[1] <= 0,
            # Missing dynamics constraints
            # Should we add 39c,39d,39e?
            # Are 39f, 39d already in here?
            self.variables["p"] >= 0,
            # add constraints for trust region
            cvx.norm(deltax, "fro")
            + cvx.norm(deltau, "fro")
            + cvx.norm(deltap, "fro")
            - self.problem_parameters["radius_trust_region"]
            <= 0,
        ]

        # Dynamic constraints
        # TODO vettorizzare usando cvx.matmul se si riesce, ma si lamenta perchè non sa gestire 3 dimensioni
        constraints += [
            self.variables["X"][:, i]
            == cvx.matmul(A[i - 1].reshape((self.n_x, self.n_x), order="F"), self.variables["X"][:, i - 1].T)
            + cvx.matmul(B_plus[i - 1].reshape((self.n_x, self.n_u), order="F"), self.variables["U"][:, i].T)
            + cvx.matmul(B_minus[i - 1].reshape((self.n_x, self.n_u), order="F"), self.variables["U"][:, i - 1].T)
            + cvx.matmul(F[i - 1].reshape((self.n_x, self.n_p), order="F"), self.variables["p"].T)
            + r[i - 1]
            + self.variables["v_dyn"][:, i - 1]
            for i in range(1, self.params.K)
        ]

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # TODO
        # Example objective
        # objective = self.params.weight_p @ self.variables["p"]
        objective = (
            cvx.norm(1 / self.params.K * self.variables["U"][0], "fro")
            + 100 * cvx.norm(self.variables["v_dyn"], "fro")
            + 100 * cvx.norm(self.variables["v_init_state"], "fro")
            + 100 * cvx.norm(self.variables["v_goal_coords"], "fro")
            + 100 * cvx.norm(self.variables["v_goal_pose"], "fro")
            + 100 * cvx.norm(self.variables["v_goal_vel"], "fro")
        )
        # add the final time
        # objective += self.params.weight_p @ self.variables["p"]
        return cvx.Minimize(objective)

    def _get_non_discretize_objective(self, delta):
        objective = cvx.norm1(1 / self.params.K * self.variables["U"][0].value) + 100 * cvx.norm1(delta)
        # objective += self.params.weight_p @ self.variables["p"].value
        objective = cvx.norm(1 / self.params.K * self.variables["U"][0].value, "fro") + 100 * cvx.norm(delta, "fro")
        objective += self.params.weight_p @ self.variables["p"].value
        ## add boundary conditions
        objective += 100 * cvx.norm(
            self.variables["X"][:, 0].value - self.problem_parameters["init_state"].value, "fro"
        )
        x_goal = self.problem_parameters["goal_config"][0].value
        y_goal = self.problem_parameters["goal_config"][1].value
        x_fin = self.variables["X"][0, self.params.K - 1].value
        y_fin = self.variables["X"][1, self.params.K - 1].value
        pose_goal = self.problem_parameters["goal_config"][2].value
        pose_fin = self.variables["X"][2, self.params.K - 1].value
        vx_fin = self.variables["X"][3, self.params.K - 1].value
        vy_fin = self.variables["X"][4, self.params.K - 1].value
        vx_goal = self.problem_parameters["goal_config"][3].value
        vy_goal = self.problem_parameters["goal_config"][4].value
        goal_coords = cvx.vstack([x_goal, y_goal])
        fin_coords = cvx.vstack([x_fin, y_fin])
        v_fin = cvx.vstack([vx_fin, vy_fin])
        v_goal = cvx.vstack([vx_goal, vy_goal])
        objective += 100 * cvx.norm(goal_coords - fin_coords, "fro")
        objective += 100 * cvx.norm(pose_fin - pose_goal, "fro")
        objective += 100 * cvx.norm(v_fin - v_goal, "fro")

        return objective.value

    def _get_objective2(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # TODO
        # Example objective
        # objective = self.params.weight_p @ self.variables["p"]
        objective = cvx.norm(1 / self.params.K * self.variables["U"], "fro")
        for i in range(self.params.K):
            objective += cvx.norm(
                1 / self.params.K * (self.variables["X"][0:6, i] - self.problem_parameters["goal_config"]), "fro"
            )
        objective += (
            100 * cvx.norm(1 / self.params.K * self.variables["v_dyn"], "fro")
            + 100 * cvx.norm(self.variables["v_init_state"], "fro")
            + 100 * cvx.norm(self.variables["v_goal_coords"], "fro")
            + 100 * cvx.norm(self.variables["v_goal_pose"], "fro")
            + 100 * cvx.norm(self.variables["v_goal_vel"], "fro")
        )

        # add the final time
        # objective += self.params.weight_p @ self.variables["p"]
        return cvx.Minimize(objective)

    def _get_non_discretize_objective2(self, delta):
        objective = cvx.norm(1 / self.params.K * self.variables["U"].value, "fro")
        for i in range(self.params.K):
            objective += cvx.norm(
                1 / self.params.K * (self.variables["X"][0:6, i].value - self.problem_parameters["goal_config"]).value,
                "fro",
            )
        objective += 100 * cvx.norm(1 / self.params.K * delta, "fro")
        objective += 100 * cvx.norm(
            self.variables["X"][:, 0].value - self.problem_parameters["init_state"].value, "fro"
        )
        x_goal = self.problem_parameters["goal_config"][0].value
        y_goal = self.problem_parameters["goal_config"][1].value
        x_fin = self.variables["X"][0, self.params.K - 1].value
        y_fin = self.variables["X"][1, self.params.K - 1].value
        pose_goal = self.problem_parameters["goal_config"][2].value
        pose_fin = self.variables["X"][2, self.params.K - 1].value
        vx_fin = self.variables["X"][3, self.params.K - 1].value
        vy_fin = self.variables["X"][4, self.params.K - 1].value
        vx_goal = self.problem_parameters["goal_config"][3].value
        vy_goal = self.problem_parameters["goal_config"][4].value
        goal_coords = cvx.vstack([x_goal, y_goal])
        fin_coords = cvx.vstack([x_fin, y_fin])
        v_fin = cvx.vstack([vx_fin, vy_fin])
        v_goal = cvx.vstack([vx_goal, vy_goal])
        objective += 100 * cvx.norm(goal_coords - fin_coords, "fro")
        objective += 100 * cvx.norm(pose_fin - pose_goal, "fro")
        objective += 100 * cvx.norm(v_fin - v_goal, "fro")

        return objective.value

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        Returns A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar from discretization around X_bar, U_bar, p.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.problem_parameters["X_bar"].value,
            self.problem_parameters["U_bar"].value,
            self.problem_parameters["p_bar"].value,
        )

        # tempA = A_bar[:, 0].reshape((8, 8), order="F")

        self.problem_parameters["init_state"].value = self.init_state.as_ndarray()
        self.problem_parameters["goal_config"].value = self.goal_state.as_ndarray()
        # TODO populate other problem parameters + function is not modifying anything
        # return (
        #     A_bar.T.reshape((-1, self.n_x, self.n_x), order="F"),
        #     B_plus_bar.T.reshape((-1, self.n_x, self.n_u), order="F"),
        #     B_minus_bar.T.reshape((-1, self.n_x, self.n_u), order="F"),
        #     F_bar.T.reshape((-1, self.n_x, self.n_p), order="F"),
        #     r_bar.T.reshape((-1, self.n_x), order="F"),
        # )
        self.problem_parameters["A_bar"].value = A_bar  # .T.reshape((-1, self.n_x, self.n_x), order="F")
        self.problem_parameters["B_plus_bar"].value = B_plus_bar  # .T.reshape((-1, self.n_x, self.n_u), order="F")
        self.problem_parameters["B_minus_bar"].value = B_minus_bar  # .T.reshape((-1, self.n_x, self.n_u), order="F")
        self.problem_parameters["F_bar"].value = F_bar  # .T.reshape((-1, self.n_x, self.n_p), order="F")
        self.problem_parameters["r_bar"].value = r_bar  # .T.reshape((-1, self.n_x), order="F")

    def _check_convergence(self, objective_pre_solver_non_discretized: float) -> bool:
        """
        Check convergence of SCvx.
        """
        return abs(self.new_objective - objective_pre_solver_non_discretized) < self.params.stop_crit * abs(
            objective_pre_solver_non_discretized
        )

    def _check_convergence2(self, X_solution, P_solution) -> bool:
        """
        Check convergence of SCvx.
        """
        return bool(
            (
                cvx.norm(X_solution - self.problem_parameters["X_bar"].value, 1)
                + cvx.norm(P_solution - self.problem_parameters["p_bar"].value, 1)
            ).value
            <= self.params.stop_crit
        )

    def _update_trust_region(self, rho: float):
        """
        Update trust region radius.
        """
        if rho < self.params.rho_1:
            self.problem_parameters["radius_trust_region"].value = max(
                self.params.min_tr_radius, self.problem_parameters["radius_trust_region"].value / self.params.alpha
            )
        elif rho >= self.params.rho_1 and rho < self.params.rho_2:
            pass
        else:
            self.problem_parameters["radius_trust_region"].value = min(
                self.params.max_tr_radius, self.params.beta * self.problem_parameters["radius_trust_region"].value
            )

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
        # assert (
        #     len(ts) == F.shape[0] == ddelta.shape[0] == npstates.shape[0]
        # ), "All inputs must have the same length as the number of timestamps."
        assert len(ts) == F.shape[0], "Length of ts must match the number of rows in F."
        assert len(ts) == ddelta.shape[0], "Length of ts must match the number of rows in ddelta."
        assert len(ts) == npstates.shape[0], "Length of ts must match the number of rows in npstates."

        # Create the command sequence
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # Create the state sequence
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)

        return mycmds, mystates


class Tester:
    """
    Class for testing the planner.
    """

    def __init__(self, planner):
        self.planner = planner

    def check_dynamics(self):
        """
        Check the dynamics between integration of A_bar, B_minus, B_plus, F_bar, r_bar and the result of
        """
        # X_bar, U_bar, p_bar = planner.initial_guess()
        U_bar = np.random.randint(-2, 2, (2, self.planner.params.K))
        p_bar = np.random.randint(0, 10, (1))
        x0 = np.random.rand(8)
        X_integrated_ground_truth = self.planner.integrator.integrate_nonlinear_full(x0, U_bar, p_bar)

        self.planner.X_bar = X_integrated_ground_truth
        self.planner.U_bar = U_bar
        self.planner.p_bar = p_bar

        print(f"X_integrated_gt shape: {X_integrated_ground_truth.shape}")
        print(f"U_bar shape: {U_bar.shape}")
        print(f"p_bar shape: {p_bar.shape}")

        A_bar, B_minus, B_plus, F_bar, r_bar = planner._convexification()
        X_res = np.zeros_like(X_integrated_ground_truth)
        X_res[:, 0] = X_integrated_ground_truth[:, 0]

        for k in range(X_integrated_ground_truth.shape[1] - 1):
            X_res[:, k + 1] = (
                A_bar[k] @ X_res[:, k]
                + B_minus[k] @ U_bar[:, k]
                + B_plus[k] @ U_bar[:, k + 1]
                + (F_bar[k] * p_bar)
                + r_bar[k]
            )
            # print(X_res[:, k + 1])
            # if X_res[7, k + 1] != 2:
            #     print(f"A_bar term: {A_bar[k] @ X_res[:, k]}")
            #     print(f"B_minus term: {B_minus[k] @ U_bar[:, k]}")
            #     print(f"B_plus term: {B_plus[k] @ U_bar[:, k + 1]}")
            #     print(f"F_bar term: {(F_bar[k] * p_bar).reshape(-1)}")
            #     print(f"r_bar term: {r_bar[k]}")

        if not np.allclose(X_res, X_integrated_ground_truth):
            print("Dynamics are not correct.")
            tol = 0.001
            indices = np.where(np.any(X_res - X_integrated_ground_truth >= tol, axis=1))[0]
            print(f"indices: {indices}")
            for i in indices:
                # print(f"X_res at {i}: {X_res[:, i]}")
                # print(f"X_integrated_GT at {i}: {X_integrated_ground_truth[:,i]}")
                print(f"Error at {i}: {X_res[:, i] - X_integrated_ground_truth[:, i]}")
        else:
            print("Dynamics are correct.")


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

    tester = Tester(planner)
    tester.check_dynamics()
    # mycmds, mystates = planner.compute_trajectory(init_state, goal_state)
