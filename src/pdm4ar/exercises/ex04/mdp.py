from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc


class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""
        self.wormholes = []
        self.trans = {}
        self.start_state = None

    def out_of_bounds(self, state: State) -> bool:
        return state[0] < 0 or state[0] >= self.grid.shape[0] or state[1] < 0 or state[1] >= self.grid.shape[1]

    def adjacent(self, next_state: State, state: State) -> bool:
        return (abs(next_state[0] - state[0]) == 1) ^ (abs(next_state[1] - state[1]) == 1)

    def allowed_actions(self, state: State) -> list[Action]:
        if self.grid[state] == 0:
            return [Action.STAY, Action.ABANDON]
        if self.grid[state] == 5:
            return [Action.ABANDON]
        l = []
        for action in Action:
            if action != Action.STAY and action != Action.ABANDON:
                if action == Action.NORTH:
                    desired_state = (state[0] - 1, state[1])
                if action == Action.SOUTH:
                    desired_state = (state[0] + 1, state[1])
                if action == Action.EAST:
                    desired_state = (state[0], state[1] + 1)
                if action == Action.WEST:
                    desired_state = (state[0], state[1] - 1)
                if not self.out_of_bounds(desired_state) and not self.grid[desired_state] == 5:
                    l.append(action)
            elif action == Action.ABANDON:
                l.append(action)
        return l

    def get_next_states(self, state: State, action: Action) -> list[tuple[State, float]]:
        next_states = []
        if action == Action.ABANDON:
            next_states.append((self.start_state, 1.0))
            return next_states
        if action == Action.NORTH:
            desired_state = (state[0] - 1, state[1])
            undesired_states = [(state[0] + 1, state[1]), (state[0], state[1] - 1), (state[0], state[1] + 1)]
        if action == Action.SOUTH:
            desired_state = (state[0] + 1, state[1])
            undesired_states = [(state[0] - 1, state[1]), (state[0], state[1] - 1), (state[0], state[1] + 1)]
        if action == Action.EAST:
            desired_state = (state[0], state[1] + 1)
            undesired_states = [(state[0] - 1, state[1]), (state[0], state[1] - 1), (state[0] + 1, state[1])]
        if action == Action.WEST:
            desired_state = (state[0], state[1] - 1)
            undesired_states = [(state[0] - 1, state[1]), (state[0] + 1, state[1]), (state[0], state[1] + 1)]

        if self.grid[state] == 1 or self.grid[state] == 2 or self.grid[state] == 4:  # start, grass, wormhole
            if (
                not self.out_of_bounds(desired_state)
                and self.grid[desired_state] != 5  # cliff
                and self.grid[desired_state] != 4  # wormhole
            ):  # normal cell, go there 0.75
                next_states.append((desired_state, 0.75))
            elif self.grid[desired_state] == 4:
                for w in self.wormholes:
                    next_states.append((w, 0.75 * 1 / len(self.wormholes)))

            for i in undesired_states:
                if not self.out_of_bounds(i) and self.grid[i] != 5 and self.grid[i] != 4:  # normal cell
                    next_states.append((i, 0.25 / 3))
                if self.out_of_bounds(i) or self.grid[i] == 5:  # cliff or out of bounds -> go to start
                    for idx, (s, prob) in enumerate(next_states):
                        if s == self.start_state:  # Sum the probability with the existing one
                            next_states[idx] = (s, prob + 0.25 / 3)
                            break
                    else:
                        next_states.append((self.start_state, 0.25 / 3))
                elif self.grid[i] == 4:
                    for w in self.wormholes:
                        for idx, (s, prob) in enumerate(next_states):
                            if s == w:  # Sum the probability with the existing one
                                next_states[idx] = (s, prob + 0.25 / 3 * 1 / len(self.wormholes))
                                break
                        else:
                            next_states.append((w, 0.25 / 3 * 1 / len(self.wormholes)))
        if self.grid[state] == 3:  # swamp
            next_states.append((state, 0.2))  # stay in swamp
            next_states.append((self.start_state, 0.05))  # break down -> go to start
            if (
                not self.out_of_bounds(desired_state)
                and self.grid[desired_state] != 5  # cliff
                and self.grid[desired_state] != 4  # wormhole
            ):  # normal cell, go there 0.5
                next_states.append((desired_state, 0.5))
            elif self.grid[desired_state] == 4:
                for w in self.wormholes:
                    next_states.append((w, 0.5 * 1 / len(self.wormholes)))

            for i in undesired_states:
                if not self.out_of_bounds(i) and self.grid[i] != 5 and self.grid[i] != 4:
                    next_states.append((i, 0.25 / 3))
                if self.out_of_bounds(i) or self.grid[i] == 5:
                    for idx, (s, prob) in enumerate(next_states):
                        if s == self.start_state:  # Sum the probability with the existing one
                            next_states[idx] = (s, prob + 0.25 / 3)
                            break
                    else:
                        next_states.append((self.start_state, 0.25 / 3))
                elif self.grid[i] == 4:
                    for w in self.wormholes:
                        for idx, (s, prob) in enumerate(next_states):
                            if s == w:  # Sum the probability with the existing one
                                next_states[idx] = (s, prob + 0.25 / 3 * 1 / len(self.wormholes))
                                break
                        else:
                            next_states.append((w, 0.25 / 3 * 1 / len(self.wormholes)))

        if self.grid[state] == 0 and action == Action.STAY:  # goal
            next_states.append((state, 1.0))

        return next_states

    def generate_transitions(self):
        if not self.wormholes:
            self.wormholes = [tuple(coord) for coord in np.argwhere(self.grid == 4).tolist()]
        if self.start_state is None:
            self.start_state = tuple(np.argwhere(self.grid == 1)[0])
        if not self.trans:
            for state in np.ndindex(self.grid.shape):
                # print(state)
                for A in self.allowed_actions(state):
                    self.trans[state, A] = self.get_next_states(state, A)
        return

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""

        # print(self.trans)
        # input("Press Enter to continue...")
        for s, p in self.trans[state, action]:
            if s == next_state:
                return p
        return 0.0

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        # todo
        if action == Action.ABANDON:
            return -10.0
        if self.grid[state] == 1 or self.grid[state] == 2 or self.grid[state] == 4:  # start, grass, wormhole
            if self.grid[next_state] == 1:
                return -11.0
            return -1.0
        if self.grid[state] == 3:  # swamp
            if self.grid[next_state] == 1:
                return -12.0
            return -2.0
        if self.grid[state] == 5:  # cliff
            return -10
        if self.grid[state] == 0:  # goal
            if action == Action.STAY:
                return 50.0

    def state_to_index(self, state: tuple[int, int]) -> int:
        return self.grid.shape[1] * state[0] + state[1]


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        pass
