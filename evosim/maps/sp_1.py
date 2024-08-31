import random
from typing import Dict, List, Tuple

import numpy as np

from evosim.elements import agents, obstacles, resources
from evosim.maps.base_map import BaseMap
from evosim.maps.cells import Pos
from evosim.utils.logger import get_logger

logger = get_logger()

MapState = Dict[str, List[List[int]]]


class SinglePlayerMap(BaseMap):

    def __init__(
        self,
        agent: agents.Agent,
        side_length: int = 32,
        obstacle_percent: float = 0.1,
        resource_percent: float = 0.2,
        waste_move_penalty: float = 1,
        death_penalty: float = 1,
        finish_reward: float = 100.0,
    ):
        super().__init__(side_length=side_length)
        self.obs_pct = obstacle_percent
        self.res_pct = resource_percent
        self.wasted_pn = waste_move_penalty
        self.death_pn = death_penalty
        self.agent = agent
        self.finish_reward = finish_reward

        self.directions = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}

    def _set_elements_on_map(self, agent: agents.Agent) -> None:
        """Scatter and place elements on map cells

        Initially agent resides in (0, 0) or top-left corner
        and objects are scatter randomly across the map
        Args:
            agent (agents.Agent): Agent of the Map
        """
        init_cell = self.fetch_cell(Pos(0, 0))
        init_cell.assign(agent)
        for idx in range(self.side_n):
            for idy in range(self.side_n):
                if idx == 0 and idy == 0:
                    continue
                else:
                    cell = self.fetch_cell(Pos(idx, idy))
                    el_probs = random.random()
                    if el_probs <= self.obs_pct:
                        obj = obstacles.Rock(Pos(idx, idy))
                        cell.assign(obj)
                    elif el_probs <= self.obs_pct + self.res_pct:
                        obj = resources.Wood(Pos(idx, idy))
                        cell.assign(obj)
        logger.info(f"Elements have been assigned to cells")

    def _get_current_state(self) -> MapState:
        """Retrieve current game state

        The current game state has 2 type of information
        The position of rocks and woods
        So, the game state will return 2 set of 2d array
        where each will define the presence of each element
        and another channel for agent

        In this scenario it'll return 2d matrix each for woods and rocks


        Returns:
            Dict[List[List]]: Game state in 3 channels
        """

        state_dict: MapState = {
            "Wood": [[0 for _ in range(self.side_n)] for _ in range(self.side_n)],
            "Rock": [[0 for _ in range(self.side_n)] for _ in range(self.side_n)],
            "Agent": [[0 for _ in range(self.side_n)] for _ in range(self.side_n)],
        }

        for idx in range(self.side_n):
            for idy in range(self.side_n):

                # defining curr position
                curr_pos = Pos(idx, idy)

                # Checking the position is occupied by any element
                if not self._is_pos_free(curr_pos):

                    # Fetching cell object
                    cell = self.fetch_cell(curr_pos)

                    # Find object type and assign to respective grid
                    obj_type = cell.placeholder.__class__.__name__

                    if "agent" in obj_type.lower():
                        state_dict["Agent"][idx][idy] = 1
                    else:
                        state_dict[obj_type][idx][idy] = 1

        return state_dict

    def _move_object(self, pos1: Pos, pos2: Pos, exists_ok: bool = False) -> None:
        """Move one object from one cell to another

        Args:
            pos1 (Pos): Current location of the object
            pos2 (Pos): Next location of the object
        """
        # Sanity before assigning
        assert not self._is_pos_free(pos1), f"{pos1} is already empty"

        # Detaching object from first cell
        cell1 = self.fetch_cell(pos1)
        cell_object = cell1.clear()

        # Moving object to new cell
        cell2 = self.fetch_cell(pos2)
        cell2.assign(cell_object, exists_ok)

    def sample(self) -> int:
        """Sample Random action

        Returns:
            int: sample action
        """
        return random.randint(0, 3)

    def reset(self) -> Tuple[MapState, bool, bool]:
        """Resets the map and set a new random initial moment

        Reset also generates a basic overview of the map objects
        i.e. observation. The observation holds 2d matrix of presence
        of each type of object
        {
            "elem1" : [[],[], ...],
            "elem2 : [[],[], ...]
        }

        Returns:
            Tuple[bool, Dict[str, List[List[int]]]]: done state and game state
        """
        self.set_map()
        self._set_elements_on_map(self.agent)
        game_state = self._get_current_state()
        terminated = False
        if np.sum(np.array(game_state["Wood"])) == 0:
            terminated = True

        return game_state, terminated, False

    def step(self, action: int) -> Tuple[MapState, int, bool, bool]:
        """Env updates upon an action

        Currently there are 5 actions possible
        0 -> move up
            Moves up if there are empty grid or grid with resources are present
        1 -> move down
            Moves down if there are empty grid or grid with resources are present
        2 -> move right
            Moves rigth if there are empty grid or grid with resources are present
        3 -> move left
            Moves left if there are empty grid or grid with resources are present
        Args:
            action (int): Action of the agent

        Raises:
            ValueError: If unknown action found

        Returns:
            Tuple[MapState, int, bool, bool]: state_dict, reward, terminated, truncated
            state_dict : Current map state
            reward : reward gained at the step
            terminated : if all the resources are finished or timeframes are finished
            truncated : if agent dies in game
        """

        terminated = False
        truncated = False
        reward = 0

        agent_pos = self.agent.pos

        # Checking if the action is valid
        if action in range(4):

            # Updated the position
            next_pos = self.agent.pos + self.directions[action]

            try:
                # fetch the corresponding cell
                cell = self.fetch_cell(next_pos)

            except:
                # If the position isn't valid, skip the step
                logger.warning(f"Not possible to move to {next_pos}. Wasted Step")
                return self._get_current_state(), reward, terminated, truncated

            # Checking if the cell is free to move the agent
            if self._is_pos_free(next_pos):

                logger.info(f"Agent is moved from {agent_pos} to {next_pos}")
                # Moved the agent to blank cell
                self._move_object(agent_pos, next_pos)
                agent_pos = next_pos
            else:

                # Checking is resources are in the cell
                if cell.c_type != "Wood":
                    logger.warning(f"{cell} can't be moved. Move wasted.")
                    reward = -self.wasted_pn
                else:
                    # Transfer resource energy to agent
                    self.agent.hp += cell.placeholder.hp
                    reward += cell.placeholder.hp

                    # Moved the agent after consuming next_pos resources
                    self._move_object(agent_pos, next_pos, True)
                    agent_pos = next_pos
                    logger.info(
                        f"Agent is moved from {agent_pos} to {next_pos} after consuming wood"
                    )

        else:
            raise ValueError(f"Action={action} not defined")

        # Agent using effort
        self.agent.run()

        # Removing dead agent
        if self.agent.hp <= 0:
            cell = self.fetch_cell(agent_pos)
            cell.clear()
            truncated = True
            reward = -self.death_pn
            logger.warning(f"Agent has died on {cell.pos}")

        # Fetch current state and check whether the game can run further
        curr_game_state = self._get_current_state()
        if np.sum(np.array(curr_game_state["Wood"])) == 0:
            terminated = True
            reward = self.finish_reward

        return curr_game_state, reward, terminated, truncated


# if __name__ == "__main__":

#     env = SinglePlayerMap()
#     obs, done = env.reset()

#     moves = 100
#     while not done and moves > 0:
#         action = env.sample()
#         obs, reward, truncated, done = env.step(action)
#         print(reward, truncated, done)
#         moves -= 1
