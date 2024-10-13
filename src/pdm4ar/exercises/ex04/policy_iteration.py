import numpy as np

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        # policy = np.zeros_like(grid_mdp.grid).astype(int)

        grid_mdp.generate_transitions()
        num_states = grid_mdp.grid.size
        num_actions = 6
        V = np.zeros(num_states).astype(float)
        I = np.eye(num_states)
        policy = np.zeros(num_states).astype(int)
        new_policy = np.zeros_like(policy)

        for state in np.ndindex(grid_mdp.grid.shape):
            state_idx = grid_mdp.state_to_index(state)
            new_policy[state_idx] = np.random.choice(grid_mdp.allowed_actions(state))

        transition_p_mat = np.zeros((num_states, num_actions, num_states)).astype(float)
        rewards_mat = np.zeros((num_states, num_actions)).astype(float)
        states_range = np.arange(num_states)  # precompute range of states for faster access
        # populate matrices
        for state in np.ndindex(grid_mdp.grid.shape):
            state_idx = grid_mdp.state_to_index(state)

            for action in grid_mdp.allowed_actions(state):
                next_states = grid_mdp.trans[state, action]

                for next_state, prob in next_states:
                    next_state_idx = grid_mdp.state_to_index(next_state)

                    transition_p_mat[state_idx, action, next_state_idx] = prob
                    rewards_mat[state_idx, action] += prob * grid_mdp.stage_reward(state, action, next_state)

        while not np.array_equal(new_policy, policy):
            policy = np.copy(new_policy)

            while True:  # Perform a few steps of value iteration
                # print(rewards_mat[states_range, policy].shape)
                # print(transition_p_mat[states_range, policy, :].shape, V.shape)
                r = rewards_mat[states_range, policy]
                t = transition_p_mat[states_range, policy, :]
                V_new = rewards_mat[states_range, policy] + grid_mdp.gamma * transition_p_mat[
                    states_range, policy, :
                ].dot(V)
                if np.max(np.abs(V - V_new)) < 1e-3:  # Check for convergence
                    break
                V = V_new
            Q = rewards_mat + grid_mdp.gamma * transition_p_mat.dot(V)
            new_policy = Q.argmax(axis=1)
            """
            # Policy Evaluation
            for state in np.ndindex(grid_mdp.grid.shape):
                action = policy[state]
                next_states = grid_mdp.trans[state, action]
                for next_state, p in next_states:
                    state_idx = grid_mdp.state_to_index(state)
                    next_state_idx = grid_mdp.state_to_index(next_state)
                    transition_p_mat[state_idx][next_state_idx] = p
                    rewards_mat[state_idx][next_state_idx] = grid_mdp.stage_reward(state, action, next_state)

            expected_reward = np.sum(transition_p_mat * rewards_mat, axis=1).reshape(-1, 1)
            v_vec = np.linalg.solve(I - grid_mdp.gamma * transition_p_mat, expected_reward)
            value_func = v_vec.reshape(grid_mdp.grid.shape)
            
            # Policy improvement
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
                        new_policy[state] = action
            """
        value_func = V.reshape(grid_mdp.grid.shape)
        policy.resize(grid_mdp.grid.shape)
        return value_func, policy
