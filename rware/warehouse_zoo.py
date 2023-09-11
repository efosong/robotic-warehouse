import logging

import functools
from collections import defaultdict, OrderedDict
from copy import copy
import gym
from gym import spaces

from gym.utils import seeding
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec

import numpy as np

from typing import List, Tuple, Optional, Dict

import networkx as nx

from .warehouse import _VectorWriter, Action, Direction, RewardType, ObservationType, Entity, Agent, Shelf, ImageLayer

_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2

_COLLISION_LAYERS = 2

_LAYER_AGENTS = 0
_LAYER_SHELFS = 1

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def parallel_env(**kwargs):
    env = WarehouseZoo(**kwargs)
    return env

def raw_env(**kwargs):
    env = parallel_env(**kwargs)
    env = parallel_to_aec(env)
    return env


class WarehouseZoo(ParallelEnv):

    metadata = {
        "name": "rware",
        "render.modes": ["human", "rgb_array"],
        "render_fps": 4,
        }

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        num_agents,
        msg_bits: int,
        sensor_range: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        reward_type: RewardType,
        layout: str = None,
        observation_type: ObservationType=ObservationType.FLATTENED,
        image_observation_layers: List[ImageLayer]=[
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE
        ],
        image_observation_directional: bool=True,
        normalised_coordinates: bool=False,
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param num_agents: Number of spawned and controlled agents
        :type num_agents: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: List[ImageLayer]
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        if not layout:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        else:
            self._make_layout_from_str(layout)

        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps
        
        self.normalised_coordinates = normalised_coordinates

        self.request_queue_size = request_queue_size
        self.request_queue = []

        self.agents = copy(self.possible_agents)
        self.agents_obj = {}

        if observation_type == ObservationType.IMAGE:
            self.image_obs = True
            self.fast_obs = False
            self.image_observation_directional = image_observation_directional
            self.image_observation_layers = image_observation_layers
        else:
            # used for DICT observation type and needed as preceeding stype to generate
            # FLATTENED observations as well
            self.image_obs = False
            self.fast_obs = (observation_type == ObservationType.FLATTENED)
            self._obs_bits_for_self = 4 + len(Direction)
            self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
            self._obs_bits_per_shelf = 2
            self._obs_bits_for_requests = 2
            self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2
            self._obs_length = (
                self._obs_bits_for_self
                + self._obs_sensor_locations * self._obs_bits_per_agent
                + self._obs_sensor_locations * self._obs_bits_per_shelf
            )

        self.renderer = None

    def _make_layout_from_params(self, shelf_columns, shelf_rows, column_height):
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"

        self.grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        self.column_height = column_height
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.goals = [
            (self.grid_size[1] // 2 - 1, self.grid_size[0] - 1),
            (self.grid_size[1] // 2, self.grid_size[0] - 1),
        ]

        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        highway_func = lambda x, y: (
            (x % 3 == 0)  # vertical highways
            or (y % (self.column_height + 1) == 0)  # horizontal highways
            or (y == self.grid_size[0] - 1)  # delivery row
            or (  # remove a box for queuing
                (y > self.grid_size[0] - (self.column_height + 3))
                and ((x == self.grid_size[1] // 2 - 1) or (x == self.grid_size[1] // 2))
            )
        )
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)

    def _make_layout_from_str(self, layout):
        layout = layout.strip()
        layout = layout.replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        self.grid_size = (grid_height, grid_width)
        self.grid = np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)
        self.highways = np.zeros(self.grid_size, dtype=np.int32)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    self.goals.append((x, y))
                    self.highways[y, x] = 1
                elif char.lower() == ".":
                    self.highways[y, x] = 1

        assert len(self.goals) >= 1, "At least one goal is required"

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    def _make_obs(self, agent_id):
        agent = self.agents_obj[agent_id]
        if self.image_obs:
            # write image observations
            if agent.id == 1:
                layers = []
                # first agent's observation --> update global observation layers
                for layer_type in self.image_observation_layers:
                    if layer_type == ImageLayer.SHELVES:
                        layer = self.grid[_LAYER_SHELFS].copy().astype(np.float32)
                        # set all occupied shelf cells to 1.0 (instead of shelf ID)
                        layer[layer > 0.0] = 1.0
                    elif layer_type == ImageLayer.REQUESTS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for requested_shelf in self.request_queue:
                            layer[requested_shelf.y, requested_shelf.x] = 1.0
                    elif layer_type == ImageLayer.AGENTS:
                        layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0
                    elif layer_type == ImageLayer.AGENT_DIRECTION:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents_obj.values():
                            agent_direction = ag.dir.value + 1
                            layer[ag.x, ag.y] = float(agent_direction)
                    elif layer_type == ImageLayer.AGENT_LOAD:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents_obj.values():
                            if ag.carrying_shelf is not None:
                                layer[ag.x, ag.y] = 1.0
                    elif layer_type == ImageLayer.GOALS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for goal_y, goal_x in self.goals:
                            layer[goal_x, goal_y] = 1.0
                    elif layer_type == ImageLayer.ACCESSIBLE:
                        layer = np.ones(self.grid_size, dtype=np.float32)
                        for ag in self.agents_obj.values():
                            layer[ag.y, ag.x] = 0.0
                    # pad with 0s for out-of-map cells
                    layer = np.pad(layer, self.sensor_range, mode="constant")
                    layers.append(layer)
                self.global_layers = np.stack(layers)

            # global information was generated --> get information for agent
            start_x = agent.y
            end_x = agent.y + 2 * self.sensor_range + 1
            start_y = agent.x
            end_y = agent.x + 2 * self.sensor_range + 1
            obs = self.global_layers[:, start_x:end_x, start_y:end_y]

            if self.image_observation_directional:
                # rotate image to be in direction of agent
                if agent.dir == Direction.DOWN:
                    # rotate by 180 degrees (clockwise)
                    obs = np.rot90(obs, k=2, axes=(1,2))
                elif agent.dir == Direction.LEFT:
                    # rotate by 90 degrees (clockwise)
                    obs = np.rot90(obs, k=3, axes=(1,2))
                elif agent.dir == Direction.RIGHT:
                    # rotate by 270 degrees (clockwise)
                    obs = np.rot90(obs, k=1, axes=(1,2))
                # no rotation needed for UP direction
            return obs

        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELFS], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELFS]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        shelfs = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)

        if self.fast_obs:
            # write flattened observations
            obs = _VectorWriter(self.observation_space(agent_id).shape[0])

            if self.normalised_coordinates:
                agent_x = agent.x / (self.grid_size[1] - 1)
                agent_y = agent.y / (self.grid_size[0] - 1)
            else:
                agent_x = agent.x
                agent_y = agent.y

            obs.write([agent_x, agent_y, int(agent.carrying_shelf is not None)])
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            obs.write(direction)
            obs.write([int(self._is_highway(agent.x, agent.y))])

            for i, (id_agent, id_shelf) in enumerate(zip(agents, shelfs)):
                if id_agent == 0:
                    obs.skip(1)
                    obs.write([1.0])
                    obs.skip(3 + self.msg_bits)
                else:
                    obs.write([1.0])
                    direction = np.zeros(4)
                    direction[self.agents_obj[self.agents[id_agent - 1]].dir.value] = 1.0
                    obs.write(direction)
                    if self.msg_bits > 0:
                        obs.write(self.agents_obj[self.agents[id_agent - 1]].message)
                if id_shelf == 0:
                    obs.skip(2)
                else:
                    obs.write(
                        [1.0, int(self.shelfs[id_shelf - 1] in self.request_queue)]
                    )

            return obs.vector
 
        # write dictionary observations
        obs = {}
        if self.normalised_coordinates:
            agent_x = agent.x / (self.grid_size[1] - 1)
            agent_y = agent.y / (self.grid_size[0] - 1)
        else:
            agent_x = agent.x
            agent_y = agent.y
        # --- self data
        obs["self"] = {
            "location": np.array([agent_x, agent_y]),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self._is_highway(agent.x, agent.y))],
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                if self.msg_bits > 0:
                    obs["sensors"][i]["local_message"] = self.msg_bits * [0]
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = self.agents_obj[self.agents[id_ - 1]].dir.value
                if self.msg_bits > 0:
                    obs["sensors"][i]["local_message"] = self.agents_obj[self.agents[id_ - 1]].message

        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelfs[id_ - 1] in self.request_queue)
                ]

        return obs

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelfs:
            self.grid[_LAYER_SHELFS, s.y, s.x] = s.id

        for a in self.agents_obj.values():
            self.grid[_LAYER_AGENTS, a.y, a.x] = a.id

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        self.seed(seed)
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # make the shelfs
        self.shelfs = [
            Shelf(x, y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y)
        ]

        # spawn agents at random locations
        agent_locs = self.np_random.choice(
            np.arange(self.grid_size[0] * self.grid_size[1]),
            size=self.max_num_agents,
            replace=False,
        )
        agent_locs = np.unravel_index(agent_locs, self.grid_size)
        # and direction
        agent_dirs = self.np_random.choice([d for d in Direction], size=self.max_num_agents)
        self.agents = copy(self.possible_agents)
        agents_obj_list = [
            Agent(x, y, dir_, self.msg_bits)
            for y, x, dir_ in zip(*agent_locs, agent_dirs)
        ]
        self.agents_obj = dict(zip(self.agents, agents_obj_list))

        self._recalc_grid()

        self.request_queue = list(
            self.np_random.choice(self.shelfs, size=self.request_queue_size, replace=False)
        )

        return {agent: self._make_obs(agent) for agent in self.agents}

    def step(self, actions):
        assert len(actions) == len(self.agents)

        for agent_id, action in actions.items():
            agent = self.agents_obj[agent_id]
            if self.msg_bits > 0:
                agent.req_action = Action(action[0])
                agent.message[:] = action[1:]
            else:
                agent.req_action = Action(action)

        # # stationary agents will certainly stay where they are
        # stationary_agents = [agent for agent in self.agents_obj if agent.action != Action.FORWARD]

        # # forward agents will move only if they avoid collisions
        # forward_agents = [agent for agent in self.agents_obj if agent.action == Action.FORWARD]
        commited_agents = set()

        G = nx.DiGraph()

        for agent in self.agents_obj.values():
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)

            if (
                agent.carrying_shelf
                and start != target
                and self.grid[_LAYER_SHELFS, target[1], target[0]]
                and not (
                    self.grid[_LAYER_AGENTS, target[1], target[0]]
                    and self.agents_obj[
                        self.agents[self.grid[_LAYER_AGENTS, target[1], target[0]] - 1]
                    ].carrying_shelf
                )
            ):
                # there's a standing shelf at the target location
                # our agent is carrying a shelf so there's no way
                # this movement can succeed. Cancel it.
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
            else:
                G.add_edge(start, target)

        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[1], start_node[0]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)
            except nx.NetworkXNoCycle:

                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(self.agents) - commited_agents

        for agent_id in failed_agents:
            agent = self.agents_obj[agent_id]
            assert agent.req_action == Action.FORWARD
            agent.req_action = Action.NOOP

        rewards = np.zeros(self.num_agents)

        for agent in self.agents_obj.values():
            agent.prev_x, agent.prev_y = agent.x, agent.y

            if agent.req_action == Action.FORWARD:
                agent.x, agent.y = agent.req_location(self.grid_size)
                if agent.carrying_shelf:
                    agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELFS, agent.y, agent.x]
                if shelf_id:
                    agent.carrying_shelf = self.shelfs[shelf_id - 1]
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                if not self._is_highway(agent.x, agent.y):
                    agent.carrying_shelf = None
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        rewards[agent.id - 1] += 0.5

                    agent.has_delivered = False

        self._recalc_grid()

        shelf_delivered = False
        for y, x in self.goals:
            shelf_id = self.grid[_LAYER_SHELFS, x, y]
            if not shelf_id:
                continue
            shelf = self.shelfs[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            # a shelf was successfully delived.
            shelf_delivered = True
            # remove from queue and replace it
            new_request = self.np_random.choice(
                list(set(self.shelfs) - set(self.request_queue))
            )
            self.request_queue[self.request_queue.index(shelf)] = new_request
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1
            elif self.reward_type == RewardType.INDIVIDUAL:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                rewards[agent_id - 1] += 1
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[_LAYER_AGENTS, x, y]
                self.agents_obj[self.agents[agent_id - 1]].has_delivered = True
                rewards[agent_id - 1] += 0.5

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        observations = {agent_id: self._make_obs(agent_id)
                        for agent_id in self.agents}
        rewards = dict(zip(self.agents, rewards))
        # NB. not 100% sure that inactivity should lead to truncation rather than termination
        terminated = False
        truncated = (
            (
                bool(self.max_inactivity_steps)
                and self._cur_inactive_steps >= self.max_inactivity_steps
            ) or (
                bool(self.max_steps)
                and self._cur_steps >= self.max_steps
            )
        )
        terminations = {agent_id: terminated for agent_id in self.agents}
        truncations = {agent_id: truncated for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self, mode="human"):
        if not self.renderer:
            from rware.rendering import Viewer

            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # depends on whether we're using image_obs, slow_obs, or fast_obs
        if self.image_obs:
            observation_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

            layers_min = []
            layers_max = []
            for layer in image_observation_layers:
                if layer == ImageLayer.AGENT_DIRECTION:
                    # directions as int
                    layer_min = np.zeros(observation_shape, dtype=np.float32)
                    layer_max = np.ones(observation_shape, dtype=np.float32) * max([d.value + 1 for d in Direction])
                else:
                    # binary layer
                    layer_min = np.zeros(observation_shape, dtype=np.float32)
                    layer_max = np.ones(observation_shape, dtype=np.float32)
                layers_min.append(layer_min)
                layers_max.append(layer_max)

            # total observation
            min_obs = np.stack(layers_min)
            max_obs = np.stack(layers_max)
            return spaces.Box(min_obs, max_obs, dtype=np.float32)
        else:
            if self.normalised_coordinates:
                location_space = spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(2,),
                        dtype=np.float32,
                )
            else:
                location_space = spaces.MultiDiscrete(
                    [self.grid_size[1], self.grid_size[0]]
                )

            self_dict = {
                "location": location_space,
                "carrying_shelf": spaces.MultiDiscrete([2]),
                "direction": spaces.Discrete(4),
                "on_highway": spaces.MultiBinary(1),
                }
            sensor_dict = {
                "has_agent": spaces.MultiBinary(1),
                "direction": spaces.Discrete(4),
                "has_shelf": spaces.MultiBinary(1),
                "shelf_requested": spaces.MultiBinary(1),
                }
            if self.msg_bits > 0:
                sensor_dict["local_message"] = spaces.MultiBinary(self.msg_bits)

            obs_space = spaces.Dict({
                "self": spaces.Dict(self_dict),
                "sensors": spaces.Tuple(
                    self._obs_sensor_locations * (spaces.Dict(sensor_dict),)
                ),
            })

            if self.fast_obs:
                return spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(spaces.flatdim(obs_space),),
                    dtype=np.float32,
                )
            else:
                return obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        action_space = [len(Action), *self.msg_bits * (2,)]
        if len(action_space) == 1:
            return spaces.Discrete(action_space[0])
        else:
            return spaces.MultiDiscrete(action_space)
    

    @property
    def observation_spaces(self):
        return {agent: self.observation_space(agent)
                for agent in self.possible_agents}

    @property
    def action_spaces(self):
        return {agent: self.action_space(agent)
                for agent in self.possible_agents}
