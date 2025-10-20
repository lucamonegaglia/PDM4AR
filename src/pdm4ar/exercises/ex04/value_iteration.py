<<<<<<< HEAD
import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc, Action
from pdm4ar.exercises_def.ex04.utils import time_function
import time
import json


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)
        # todo implement here
        convergence = np.zeros_like(grid_mdp.grid).astype(bool)
        grid_mdp.generate_transitions()
        """with open("output.txt", "w") as file:
            data_str_keys = {str(k): str(v) for k, v in grid_mdp.trans.items()}
            json.dump(data_str_keys, file, indent=4)
            print("dumpino fatto")
        
        # Verify solution
        gt_value_func = np.zeros_like(grid_mdp.grid).astype(float)
        gt_value_func[(2, 4)] = 500
        gt_value_func[(0, 2)] = 200.2
        gt_value_func[(1, 1)] = 203.5
        gt_value_func[(1, 2)] = 225.6
        gt_value_func[(1, 3)] = 214.9
        gt_value_func[(2, 0)] = 202.8
        gt_value_func[(2, 1)] = 229.5
        gt_value_func[(2, 2)] = 228.4
        gt_value_func[(2, 3)] = 385.9
        gt_value_func[(3, 1)] = 203.5
        gt_value_func[(3, 2)] = 225.6
        gt_value_func[(3, 3)] = 214.9
        gt_value_func[(4, 2)] = 200.2
        for state in np.ndindex(grid_mdp.grid.shape):
            max_expected_value = -np.inf
            for action in grid_mdp.allowed_actions(state):
                next_states = grid_mdp.trans[state, action]
                expected_value = 0
                for next_state, p in next_states:
                    expected_value += p * (
                        grid_mdp.stage_reward(state, action, next_state) + grid_mdp.gamma * gt_value_func[next_state]
                    )
                    if state == (2, 3) and action == Action.EAST:
                        print(next_state, grid_mdp.stage_reward(state, action, next_state))
                if expected_value > max_expected_value:
                    max_expected_value = expected_value
            if abs(max_expected_value - gt_value_func[state]) >= 0.1 and grid_mdp.grid[state] != 5:
                print("HAISBAGLIATOCAZZOOOO", state, max_expected_value, gt_value_func[state])
        """
        delta_v = np.inf
        epsilon = 0.01
        while delta_v > epsilon:
            delta_v = 0
            for state in np.ndindex(grid_mdp.grid.shape):
                max_expected_value = -np.inf

                for action in grid_mdp.allowed_actions(state):
                    next_states = grid_mdp.trans[state, action]
                    expected_value = 0
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
=======
import numpy as np
from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import Policy, ValueFunc
from pdm4ar.exercises_def.ex04.utils import time_function


class ValueIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)

        # todo implement here

        return value_func, policy
>>>>>>> ex11/master
