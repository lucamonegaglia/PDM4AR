import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc, Action
from pdm4ar.exercises_def.ex04.utils import time_function
import time


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)
        # todo implement here
        convergence = np.zeros_like(grid_mdp.grid).astype(bool)
        grid_mdp.generate_transitions()
        delta_v = np.inf
        epsilon = 1e-6
        while delta_v > epsilon:
            delta_v = 0
            for state in np.ndindex(grid_mdp.grid.shape):
                max_expected_value = -np.inf

                for action in grid_mdp.allowed_actions(state):
                    next_states = grid_mdp.trans[state, action]
                    expected_value = 0
                    if state == (0, 2) and value_func[state] > 198:
                        print("here")
                    for next_state, p in next_states:
                        expected_value += p * (
                            grid_mdp.stage_reward(state, action, next_state) + grid_mdp.gamma * value_func[next_state]
                        )
                    if expected_value > max_expected_value:
                        max_expected_value = expected_value
                        policy[state] = action
                # print("Ex time: {}".format(end_time - start_time))
                delta_v = max(delta_v, abs(value_func[state] - max_expected_value))
                value_func[state] = max_expected_value

        return value_func, policy
