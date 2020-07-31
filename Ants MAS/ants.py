import random
import time
import pickle
import numpy as np
import pygame as pg
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# pygame places images by top left coordinate not by center coords of image

SCREEN_SIZE = (1000, 1000)
ENV_SIZE = (100, 100)
scale = SCREEN_SIZE[0] // ENV_SIZE[0]

# 100x100 env --> 10x10 states q_table
dim_compression_coef = 10
n_states = (ENV_SIZE[0] // dim_compression_coef) * (ENV_SIZE[1] // dim_compression_coef)
n_actions = 8

# draw env_map
draw_env_window = True
raw_map_view = False
show_pheromone_threshold = 0.0
include_leaders = False
load_qt = False
learn_qt = True
if load_qt:
    with open("ants.pkl", 'rb') as f:
        q_table = pickle.load(f)
else:
    q_table = np.ones([n_states * 2, n_actions])

Q_TABLE_INITIAL_VALUE = q_table[0, 0]
EXPLORING_RATE = 0.01

n_episodes = 80000
EPSILON = 0.999
EPSILON_START_DECAY = 15000
EPSILON_DECAY = 1 / (n_episodes - EPSILON_START_DECAY) * 10
FOLLOWING_PROBABILITY = 0.1
alpha = 0.2
gamma = 0.95

ANT_SIZE = (20, 20)
ANTS_NUMBER = 100

ANT_LEADER_SIZE = (40, 40)
ANT_LEADERS_NUMBER = 4  # not more than num_of_resources

RESOURCE_SIZE = (120, 120)
RESOURCE_DIAMETER = 10  # do not go beyond screen
NUMBER_OF_RESOURCES = 4  # also edit res_images if changing this
RESOURCE_AMOUNT = 25000
RESOURCE_LOCATIONS = [(80, 0),  # top right
                      (80, 40),  # middle right
                      (10, 70),  # middle left
                      (40, 80)]  # middle bottom
CELLS = {
    'obstacle': 0,
    'empty': 1,
    'pheromone': 2,
    'resource': 3,
    'anthill': 4
}

ANTHILL_SIZE = (150, 150)
ANTHILL_DIAMETER = 10  # do not go beyond screen
ANTHILL_POS = (0, 0)

LEADER_PHEROMONE_SECRETION_RATE = 0.05
PHEROMONE_SECRETION_RATE = 0.0005
PHEROMONE_EVAPORATION_RATE = 0.0001

x_scaled = SCREEN_SIZE[0] // ENV_SIZE[0]
y_scaled = SCREEN_SIZE[1] // ENV_SIZE[1]

# colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
fancy = (150, 0, 255)
yellow = (255, 255, 0)


class Environment:
    def __init__(self, resources_list):
        self.res_list = resources_list
        self.env_size = ENV_SIZE
        self.env_tick = 0
        self.anthill_pos = ANTHILL_POS
        self.cells = CELLS
        self.resource_locations = RESOURCE_LOCATIONS
        self.num_of_resources = NUMBER_OF_RESOURCES
        self.resources_left = NUMBER_OF_RESOURCES
        self.collected_res_count = 0
        self.place_resources(self.res_list)

        self.obstacles = [(100, 20), (99, 20), (98, 20), (97, 20), (96, 20), (95, 20),  # right
                          (94, 20), (93, 20), (92, 20), (91, 20), (91, 21), (91, 22),
                          (91, 23), (91, 24), (91, 25), (91, 26), (91, 27), (91, 28),
                          (91, 29), (91, 30), (91, 31), (91, 32), (91, 33), (91, 34),
                          (91, 35), (91, 36), (91, 37), (91, 38), (91, 39), (91, 40),
                          (90, 40), (89, 40), (88, 40), (87, 40), (86, 40), (85, 40),
                          (84, 40), (83, 40), (82, 40), (81, 40), (80, 40), (80, 41),
                          (80, 42), (80, 43), (80, 44), (80, 45), (80, 46), (80, 47),
                          (80, 48), (80, 49), (80, 50), (80, 51), (80, 52), (80, 53),
                          (80, 54), (80, 55), (80, 56), (80, 57), (80, 58), (80, 59),
                          (80, 60), (81, 60), (82, 60), (83, 60), (84, 60), (85, 60),
                          (86, 60), (87, 60), (88, 60), (89, 60), (90, 60), (79, 60),
                          (78, 60), (77, 60), (76, 60), (75, 60), (74, 60), (73, 60),
                          (72, 60), (71, 60), (71, 60), (76, 60), (76, 60), (76, 60),
                          (79, 40), (78, 40), (77, 40), (76, 40), (75, 40), (74, 40),
                          (73, 40), (72, 40), (71, 40),
                          (100, 24), (99, 25), (98, 26), (97, 27), (96, 28), (95, 29),
                          (94, 30), (93, 31), (92, 32), (91, 33),
                          (99, 24), (98, 25), (97, 26), (96, 27), (95, 28), (94, 29),
                          (93, 30), (92, 31), (91, 32),
                          (71, 40), (71, 41), (71, 42), (71, 43), (71, 44), (71, 45),
                          (71, 46), (71, 47), (71, 48), (71, 49), (71, 50), (71, 51),
                          (71, 52), (71, 53), (71, 54), (71, 55), (71, 56), (71, 57),
                          (71, 58), (71, 59), (71, 60), (91, 60),

                          (70, 100), (70, 99), (70, 98), (70, 97), (70, 96), (70, 95),  # bottom
                          (70, 94), (70, 93), (70, 92), (70, 91),
                          (60, 90), (60, 89), (60, 88), (60, 87), (60, 86), (60, 85),
                          (60, 84), (60, 83), (60, 82), (60, 81),
                          (48, 80), (49, 80), (21, 80), (47, 80), (46, 80),
                          (45, 80), (44, 80), (43, 80), (42, 80), (41, 80), (40, 80),
                          (39, 80), (38, 80), (37, 80), (36, 80), (35, 80), (34, 80),
                          (33, 80), (32, 80), (31, 80), (30, 80), (29, 80), (28, 80),
                          (27, 80), (26, 80), (25, 80), (24, 80), (23, 80), (22, 80),
                          (49, 80), (50, 80), (51, 80), (52, 80), (53, 80), (54, 80),
                          (55, 80), (56, 80), (57, 80), (58, 80), (59, 80), (60, 80),
                          (48, 70), (49, 70), (21, 70), (47, 70), (46, 70), (21, 70),
                          (45, 70), (44, 70), (43, 70), (42, 70), (41, 70), (40, 70),
                          (39, 70), (38, 70), (37, 70), (36, 70), (35, 70), (34, 70),
                          (33, 70), (32, 70), (31, 70), (30, 70), (29, 70), (28, 70),
                          (27, 70), (26, 70), (25, 70), (24, 70), (23, 70), (22, 70),
                          (49, 70), (50, 70), (51, 70), (52, 70), (53, 70), (54, 70),
                          (55, 70), (56, 70), (57, 70), (58, 70), (59, 70), (60, 70),
                          (61, 90), (62, 90), (63, 90), (64, 90),
                          (65, 90), (66, 90), (67, 90), (68, 90), (69, 90), (70, 90),
                          (21, 80), (21, 79), (21, 78), (21, 77), (21, 76), (21, 75),
                          (21, 74), (21, 73), (21, 72), (21, 71), (21, 70),
                          (21, 69), (21, 68), (21, 67), (21, 66), (21, 65), (21, 64),
                          (21, 63), (21, 62), (21, 61), (21, 60), (21, 59), (21, 58),
                          (21, 57), (21, 56), (21, 55), (21, 54), (21, 53), (21, 52),
                          (21, 80), (21, 51), (60, 79), (60, 78), (60, 77), (60, 76),
                          (60, 75), (60, 74), (60, 73), (60, 72), (60, 71), (60, 79),

                          (80, 1), (80, 2), (80, 3), (80, 4), (80, 5),  # top
                          (80, 6), (80, 7), (80, 8), (80, 9), (80, 10), (80, 11),
                          (80, 12), (80, 13), (80, 14), (80, 15), (80, 16), (80, 17),
                          (80, 18), (80, 19), (80, 20), (79, 20), (78, 20), (77, 20),
                          (76, 20), (75, 20), (74, 20), (73, 20), (72, 20), (71, 20),
                          (70, 21), (70, 22), (70, 23), (70, 24), (70, 25), (70, 26),
                          (70, 20), (67, 30), (66, 30), (65, 30), (64, 30), (63, 30),
                          (70, 27), (68, 30), (69, 30), (70, 30), (60, 28), (60, 29),
                          (70, 28), (70, 29),
                          (62, 30), (61, 30), (60, 30), (60, 27), (60, 26), (60, 25),
                          (60, 24), (60, 23), (60, 22), (60, 21), (60, 20), (60, 19),
                          (60, 18), (60, 17), (60, 16), (59, 16), (58, 16), (57, 16),
                          (56, 16), (55, 16), (54, 16), (53, 16), (52, 16), (51, 16),
                          (50, 16), (49, 16), (48, 16), (47, 16), (46, 16), (45, 16),
                          (44, 16), (43, 16), (42, 16), (41, 16), (40, 16), (40, 16),
                          (40, 15), (40, 14), (40, 13), (40, 12), (40, 11), (40, 10),
                          (40, 9), (40, 8), (40, 7), (40, 6), (40, 5), (40, 4),
                          (40, 3), (40, 2), (40, 1)]
        self.env_map = self.build_env(self.res_list)
        self.home_pheromone_map = np.zeros(ENV_SIZE)
        self.explore_pheromone_map = np.zeros(ENV_SIZE)
        self.resource_area = self.define_resource_area()
        self.anthill_area = self.define_anthill_area()

    def increment_env_tick(self):
        self.env_tick += 1

    def collected_counter_increment(self):
        self.collected_res_count += 1

    def define_resource_area(self):
        area = []
        for x in range(ENV_SIZE[0]):
            for y in range(ENV_SIZE[1]):
                if self.env_map[x][y] == self.cells['resource']:
                    area.append((x, y))
        return area

    def define_anthill_area(self):
        area = []
        for x in range(ENV_SIZE[0]):
            for y in range(ENV_SIZE[1]):
                if self.env_map[x][y] == self.cells['anthill']:
                    area.append((x, y))
        return area

    def place_resources(self, res_list):
        random_loc_sample = random.sample(range(len(self.resource_locations)),
                                          self.num_of_resources)
        for res in res_list:
            res.set_position(self.resource_locations[random_loc_sample[res.id]])

    def build_env(self, res_list):
        environment = np.full(self.env_size, self.cells['empty'])

        for x in range(self.anthill_pos[0], self.anthill_pos[0] + ANTHILL_DIAMETER):
            for y in range(self.anthill_pos[1], self.anthill_pos[1] + ANTHILL_DIAMETER):
                environment[x][y] = self.cells['anthill']

        for res in res_list:
            for x in range(res.position[0], res.position[0] + RESOURCE_DIAMETER):
                for y in range(res.position[1], res.position[1] + RESOURCE_DIAMETER):
                    environment[x][y] = self.cells['resource']

        for block in self.obstacles:
            environment[block[0] - 1][block[1] - 1] = self.cells['obstacle']
        return environment

    def update_home_pheromone_map(self, position):
        if self.home_pheromone_map[position[1]][position[0]] < 1:
            self.home_pheromone_map[position[1]][position[0]] += PHEROMONE_SECRETION_RATE

        if self.home_pheromone_map[position[1]][position[0]] > 1:
            self.home_pheromone_map[position[1]][position[0]] = 1

    def update_explore_pheromone_map(self, position):
        if self.explore_pheromone_map[position[1]][position[0]] < 1:
            self.explore_pheromone_map[position[1]][position[0]] += PHEROMONE_SECRETION_RATE

        if self.explore_pheromone_map[position[1]][position[0]] > 1:
            self.explore_pheromone_map[position[1]][position[0]] = 1

    def evaporation(self):
        for x in range(ENV_SIZE[0]):
            for y in range(ENV_SIZE[1]):
                if self.home_pheromone_map[x][y] > 0:
                    self.home_pheromone_map[x][y] -= PHEROMONE_EVAPORATION_RATE
                else:
                    self.home_pheromone_map[x][y] = 0

                if self.home_pheromone_map[x][y] < 0:
                    self.home_pheromone_map[x][y] = 0

                if self.explore_pheromone_map[x][y] > 0:
                    self.explore_pheromone_map[x][y] -= PHEROMONE_EVAPORATION_RATE
                else:
                    self.explore_pheromone_map[x][y] = 0

                if self.explore_pheromone_map[x][y] < 0:
                    self.explore_pheromone_map[x][y] = 0

    def ant_leader_home_pheromone_secretion(self, position):
        if self.home_pheromone_map[position[1]][position[0]] < 1:
            self.home_pheromone_map[position[1]][position[0]] += LEADER_PHEROMONE_SECRETION_RATE

        if self.home_pheromone_map[position[1]][position[0]] > 1:
            self.home_pheromone_map[position[1]][position[0]] = 1

    def ant_leader_explore_pheromone_secretion(self, position):
        if self.explore_pheromone_map[position[1]][position[0]] < 1:
            self.explore_pheromone_map[position[1]][position[0]] += LEADER_PHEROMONE_SECRETION_RATE

        if self.explore_pheromone_map[position[1]][position[0]] > 1:
            self.explore_pheromone_map[position[1]][position[0]] = 1

    def get_home_pheromone_map(self):
        return self.home_pheromone_map

    def get_explore_pheromone_map(self):
        return self.explore_pheromone_map

    def take_resource(self, position):
        for res in self.res_list:
            for x in range(res.get_position()[0], res.get_position()[0] + RESOURCE_DIAMETER):
                for y in range(res.get_position()[1], res.get_position()[1] + RESOURCE_DIAMETER):
                    if position == (x, y):
                        if res.amount > 0:
                            res.decrement_amount()
                            return True
                        else:
                            return False

    def decrement_resource_quantity(self):
        self.resources_left -= 1

    def update_resource_area(self):
        for res in self.res_list:
            if res.amount < 1:
                for x in range(res.get_position()[0], res.get_position()[0] + RESOURCE_DIAMETER):
                    for y in range(res.get_position()[1], res.get_position()[1] + RESOURCE_DIAMETER):
                        self.env_map[x][y] = self.cells['empty']

                self.resource_area = self.define_resource_area()


class Resource:
    def __init__(self, id, coords, image):
        self.id = id
        self.position = coords
        self.image = image
        self.amount = RESOURCE_AMOUNT

    def set_position(self, new_coords):
        self.position = new_coords

    def get_position(self):
        return self.position

    def decrement_amount(self):
        self.amount -= 1


class Agent:
    def __init__(self, id, coords, map):
        self.id = id
        self.position = coords
        self.env_map = map
        self.has_resource = False
        self.epsilon = EPSILON
        self.total_reward = 0
        self.cycle_total_reward = 0
        self.mine_cycle_ended = False
        self.in_same_state = 0
        self.resources_left = NUMBER_OF_RESOURCES

    def step(self, env):
        if self.mine_cycle_ended:
            self.cycle_total_reward = self.total_reward
            self.total_reward = 0
            self.mine_cycle_ended = False

        state = self.get_state()
        actions = np.nonzero(self.available_actions())[0]
        action = self.choose_action(state, actions, env)

        if not self.has_resource:
            env.update_explore_pheromone_map(self.position)
        else:
            env.update_home_pheromone_map(self.position)

        self.move(action)
        new_state = self.get_state()

        if state == new_state:
            self.in_same_state += 1
        else:
            self.in_same_state = 0

        if (self.epsilon < (2 * EPSILON / 3)) and (self.in_same_state > 300):
            self.epsilon = EPSILON

        if self.in_same_state > 600:
            self.has_resource = False
            self.position = ANTHILL_POS

        if self.resources_left > env.resources_left:
            self.epsilon = 0.6
            self.resources_left = env.resources_left

        resource_area = env.resource_area
        anthill_area = env.anthill_area

        reward = -1  # step penalty
        if not self.has_resource and (self.position in resource_area):
            if env.take_resource(self.position):
                self.has_resource = True
                reward += 100
                # print('Resource!')
            else:
                env.decrement_resource_quantity()
                env.update_resource_area()
        if self.has_resource and (self.position in anthill_area):
            self.has_resource = False
            reward += 100
            env.collected_counter_increment()
            self.mine_cycle_ended = True
            # print('Home!')

        self.total_reward += reward

        if learn_qt:
            if state != new_state:
                self.learn(state, new_state, reward, action)

        if (self.epsilon > 0) and (env.env_tick > EPSILON_START_DECAY):
            self.epsilon -= EPSILON_DECAY

    def move(self, direction):
        dx, dy = 0, 0
        if direction == 0:
            dy = -1
        elif direction == 1:
            dy = -1
            dx = 1
        elif direction == 2:
            dx = 1
        elif direction == 3:
            dy = 1
            dx = 1
        elif direction == 4:
            dy = 1
        elif direction == 5:
            dy = 1
            dx = -1
        elif direction == 6:
            dx = -1
        elif direction == 7:
            dy = -1
            dx = -1

        x, y = self.position[0], self.position[1]
        x += dx
        y += dy

        if x < 0:
            x = 0
        if x > len(self.env_map[0]):
            x = len(self.env_map[0]) - 1
        if y < 0:
            y = 0
        if y > len(self.env_map[1]):
            y = len(self.env_map[1]) - 1

        self.position = x, y

    def available_actions(self):
        act0, act1, act2, act3 = 0, 0, 0, 0
        act4, act5, act6, act7 = 0, 0, 0, 0
        up = True
        right = True
        down = True
        left = True

        if (self.position[0] - 1) < 0:
            left = False
        if (self.position[0] + 1) > (len(self.env_map[0]) - 1):
            right = False
        if (self.position[1] - 1) < 0:
            up = False
        if (self.position[1] + 1) > (len(self.env_map[1]) - 1):
            down = False

        block = CELLS['obstacle']

        if up and not (self.env_map[self.position[0]][self.position[1] - 1] == block):
            act0 = 1
        if up and right and not (self.env_map[self.position[0] + 1][self.position[1] - 1] == block):
            act1 = 1
        if right and not (self.env_map[self.position[0] + 1][self.position[1]] == block):
            act2 = 1
        if right and down and not (self.env_map[self.position[0] + 1][self.position[1] + 1] == block):
            act3 = 1
        if down and not (self.env_map[self.position[0]][self.position[1] + 1] == block):
            act4 = 1
        if left and down and not (self.env_map[self.position[0] - 1][self.position[1] + 1] == block):
            act5 = 1
        if left and not (self.env_map[self.position[0] - 1][self.position[1]] == block):
            act6 = 1
        if left and up and not (self.env_map[self.position[0] - 1][self.position[1] - 1] == block):
            act7 = 1
        return [act0, act1, act2, act3, act4, act5, act6, act7]

    def get_state(self, position=None):
        if not position:
            if not self.has_resource:
                state = (self.position[0] // (ENV_SIZE[0] // dim_compression_coef)) * dim_compression_coef + \
                        (self.position[1] // (ENV_SIZE[1] // dim_compression_coef))
            else:
                state = (self.position[0] // (ENV_SIZE[0] // dim_compression_coef)) * dim_compression_coef + \
                        (self.position[1] // (ENV_SIZE[1] // dim_compression_coef)) + \
                        q_table.shape[0] // 2 - 1
        else:
            if not self.has_resource:
                state = (position[0] // (ENV_SIZE[0] // dim_compression_coef)) * dim_compression_coef + \
                        (position[1] // (ENV_SIZE[1] // dim_compression_coef))
            else:
                state = (position[0] // (ENV_SIZE[0] // dim_compression_coef)) * dim_compression_coef + \
                        (position[1] // (ENV_SIZE[1] // dim_compression_coef)) + \
                        q_table.shape[0] // 2 - 1

        return state

    def choose_action(self, state, avail_actions_ind, env):
        if random.uniform(0, 1) < self.epsilon:
            # exploring unknown area
            if random.uniform(0, 1) < EXPLORING_RATE:
                for act_ind in range(len(avail_actions_ind)):
                    keys = np.arange(len(avail_actions_ind))
                    act_ind_decode = dict(zip(keys, avail_actions_ind))

                    # checking area around agent
                    delta_x, delta_y = self.try_move(act_ind_decode[act_ind])
                    x, y = self.position[0] + delta_x, self.position[1] + delta_y

                    if q_table[self.get_state((x, y)), act_ind] == Q_TABLE_INITIAL_VALUE:
                        return act_ind_decode[act_ind]

            action = np.random.choice(avail_actions_ind)
        else:
            qt_arr = np.zeros(len(avail_actions_ind))
            keys = np.arange(len(avail_actions_ind))
            act_ind_decode = dict(zip(keys, avail_actions_ind))

            if random.uniform(0, 1) < FOLLOWING_PROBABILITY:
                if self.has_resource:
                    pheromone_map = env.get_home_pheromone_map()
                else:
                    pheromone_map = env.get_explore_pheromone_map()
                for act_ind in range(len(avail_actions_ind)):
                    # checking area around agent
                    delta_x, delta_y = self.try_move(act_ind_decode[act_ind])
                    x, y = self.position[0] + delta_x, self.position[1] + delta_y
                    qt_arr[act_ind] = q_table[state, act_ind_decode[act_ind]] * pheromone_map[x][y]
            else:
                for act_ind in range(len(avail_actions_ind)):
                    qt_arr[act_ind] = q_table[state, act_ind_decode[act_ind]]

            action = act_ind_decode[np.argmax(qt_arr)]
        return action

    def try_move(self, direction):
        dx, dy = 0, 0
        if direction == 0:
            dy = -1
        elif direction == 1:
            dy = -1
            dx = 1
        elif direction == 2:
            dx = 1
        elif direction == 3:
            dy = 1
            dx = 1
        elif direction == 4:
            dy = 1
        elif direction == 5:
            dy = 1
            dx = -1
        elif direction == 6:
            dx = -1
        elif direction == 7:
            dy = -1
            dx = -1
        return dx, dy

    def learn(self, state, state2, reward, action):
        q_table[state, action] = q_table[state, action] + alpha * \
                                 (reward + gamma * np.max(q_table[state2, :]) - q_table[state, action])


class AgentLeader:
    def __init__(self, id, coords, map, res_position):
        self.id = id
        self.position = coords
        self.env_map = map
        self.res_position = res_position
        self.has_resource = False
        self.path = self.find_path(ANTHILL_POS, res_position)
        self.path_stage = 0

    def find_path(self, coords, new_coords):
        matrix = np.array(self.env_map).T
        grid = Grid(matrix=matrix)
        start = grid.node(coords[0], coords[1])
        end = grid.node(new_coords[0], new_coords[1])
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, _ = finder.find_path(start, end, grid)
        return path

    def move(self, env):
        if self.position == self.res_position:
            self.has_resource = True

        if not self.has_resource:
            self.path_stage += 1
            env.ant_leader_explore_pheromone_secretion(self.position)
            self.position = self.path[self.path_stage]
        else:
            self.path_stage -= 1
            env.ant_leader_home_pheromone_secretion(self.position)
            self.position = self.path[self.path_stage]

        if self.position == ANTHILL_POS:
            self.has_resource = False


class Ant:
    def __init__(self, coords, image, screen_size):
        self.coords = coords
        self.image = image
        self.turned_image = image
        self.screen_size = screen_size
        self.angle = 0

    def move(self, new_coords):
        x, y = self.coords
        new_x, new_y = new_coords

        if (x, y) != (new_x, new_y):
            self.angle = 270 - np.rad2deg(np.arctan2(new_y - y, new_x - x))

        self.turned_image = pg.transform.rotate(self.image, self.angle)

        if new_x >= self.screen_size[0] - self.image.get_rect().size[0]:
            new_x = self.screen_size[0] - 1 - self.image.get_rect().size[0]
        if new_y >= self.screen_size[1] - self.image.get_rect().size[1]:
            new_y = self.screen_size[1] - 1 - self.image.get_rect().size[1]
        if new_x < 0:
            new_x = 0
        if new_y < 0:
            new_y = 0

        self.coords = (new_x, new_y)

    def get_position(self):
        return self.coords

    def get_turned_image(self):
        return self.turned_image


class AntLeader(Ant):
    def __init__(self, coords, image, screen_size):
        super().__init__(coords, image, screen_size)


def draw_env():
    screen.blit(lawn_image, (0, 0))
    screen.blit(anthill_image, (ANTHILL_POS[0] * x_scaled - ANTHILL_SIZE[0] // 6,
                                ANTHILL_POS[1] * y_scaled - ANTHILL_SIZE[1] // 6))

    for res in resources:
        if res.amount > 0:
            screen.blit(res.image, (res.position[0] * x_scaled - RESOURCE_SIZE[0] // 8,
                                    res.position[1] * y_scaled - RESOURCE_SIZE[1] // 8))

    font = pg.font.SysFont(None, 4 * scale)
    mean_reward = sum(mean_total_reward) / len(mean_total_reward)
    screen.blit(font.render('Step ' + str(env.env_tick), True, black), (45 * x_scaled,
                                                                        1 * y_scaled))
    screen.blit(font.render('Collected resources: ' + str(env.collected_res_count), True, black),
                (45 * x_scaled, 4 * y_scaled))
    screen.blit(font.render('Mean reward per', True, black),
                (45 * x_scaled, 7 * y_scaled))
    screen.blit(font.render('mining cycle: ' + str(mean_reward), True, black),
                (45 * x_scaled, 10 * y_scaled))

    if raw_map_view:
        for x in range(ENV_SIZE[0]):
            for y in range(ENV_SIZE[1]):
                if env.env_map[x][y] == CELLS['obstacle']:  # obstacle
                    pg.draw.rect(screen, black, [x * x_scaled, y * y_scaled, scale, scale])
                elif env.env_map[x][y] == CELLS['resource']:  # resource
                    pg.draw.rect(screen, green, [x * x_scaled, y * y_scaled, scale, scale])
                elif env.env_map[x][y] == CELLS['anthill']:  # anthill
                    pg.draw.rect(screen, white, [x * x_scaled, y * y_scaled, scale, scale])

                if not (x % 10) and not (y % 10):
                    pg.draw.rect(screen, fancy, [x * x_scaled, y * y_scaled, scale, scale])


def draw_transparent_pheromone(color, size):
    surface = pg.Surface((size, size), pg.SRCALPHA)
    pg.draw.rect(surface, color, surface.get_rect())
    return surface


def draw_res_count(count, coords, color):
    font = pg.font.SysFont(None, 40)
    text = font.render(str(count), True, color, black)
    screen.blit(text, coords)


lawn_image = pg.image.load('lawn.png')
lawn_image = pg.transform.scale(lawn_image, SCREEN_SIZE)

anthill_image = pg.image.load('anthill.png')
anthill_image = pg.transform.scale(anthill_image, ANTHILL_SIZE)

ant_image = pg.image.load('ant.png')
ant_image = pg.transform.scale(ant_image, ANT_SIZE)

ant_leader_image = pg.image.load('ant_leader.png')
ant_leader_image = pg.transform.scale(ant_leader_image, ANT_LEADER_SIZE)

berry_image = pg.image.load('blueberry.png')
berry_image = pg.transform.scale(berry_image, RESOURCE_SIZE)

mineral_image = pg.image.load('quartz.png')
mineral_image = pg.transform.scale(mineral_image, RESOURCE_SIZE)

res_images = [berry_image, mineral_image, berry_image, mineral_image]

ant_images = []
ant_agents = []

ant_leader_images = []
ant_leader_agents = []

resources = []

for res in range(NUMBER_OF_RESOURCES):
    resources.append(Resource(res, (0, 0), res_images[res]))

env = Environment(resources)

for ant in range(ANTS_NUMBER):
    ant_images.append(Ant((ANTHILL_POS[0] * x_scaled, ANTHILL_POS[1] * y_scaled),
                          ant_image, SCREEN_SIZE))
    ant_agents.append(Agent(ant, (ANTHILL_POS[0], ANTHILL_POS[1]), env.env_map))

for leader in range(ANT_LEADERS_NUMBER):
    ant_leader_images.append(AntLeader((ANTHILL_POS[0] * x_scaled, ANTHILL_POS[1] * y_scaled),
                                       ant_leader_image, SCREEN_SIZE))
    ant_leader_agents.append(AgentLeader(leader, (ANTHILL_POS[0], ANTHILL_POS[1]), env.env_map,
                                         resources[leader].position))

if __name__ == "__main__":
    pg.init()
    screen = pg.display.set_mode(SCREEN_SIZE)
    pg.display.set_caption('Ant Colony Q-learning')
    clock = pg.time.Clock()
    start = time.time()

    while (env.env_tick < n_episodes) and (env.resources_left > 0):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pass

        mean_total_reward = []
        mean_epsilon = []

        if include_leaders:
            for i in range(len(ant_leader_agents)):
                ant_leader_agents[i].move(env)

        for ant in ant_agents:
            ant.step(env)
            mean_total_reward.append(ant.cycle_total_reward)
            mean_epsilon.append(ant.epsilon)

        if draw_env_window:
            draw_env()

            for x in range(ENV_SIZE[0]):
                for y in range(ENV_SIZE[1]):
                    if env.explore_pheromone_map[x][y] > show_pheromone_threshold:
                        transparency_level = env.explore_pheromone_map[x][y] * 200
                        pheromone = draw_transparent_pheromone(blue + (transparency_level,), scale)
                        screen.blit(pheromone, (y * y_scaled, x * x_scaled))
                    if env.home_pheromone_map[x][y] > show_pheromone_threshold:
                        transparency_level = env.home_pheromone_map[x][y] * 200
                        pheromone = draw_transparent_pheromone(red + (transparency_level,), scale)
                        screen.blit(pheromone, (y * y_scaled, x * x_scaled))

            for agent in range(ANTS_NUMBER):
                position = ant_agents[agent].position
                scaled_position = (position[0] * x_scaled, position[1] * y_scaled)
                ant_images[agent].move(scaled_position)
                screen.blit(ant_images[agent].get_turned_image(),
                            (ant_images[agent].get_position()[0] - ANT_SIZE[0] // 2,
                             ant_images[agent].get_position()[1] - ANT_SIZE[1] // 2))

            for res in resources:
                if res.amount > (RESOURCE_AMOUNT / 2):
                    res_count_color = green
                elif (RESOURCE_AMOUNT / 10) <= res.amount <= (RESOURCE_AMOUNT / 2):
                    res_count_color = yellow
                else:
                    res_count_color = red
                draw_res_count(res.amount, (res.position[0] * x_scaled + RESOURCE_SIZE[0] // 12,
                                            res.position[1] * y_scaled + RESOURCE_SIZE[1] // 6 * 5),
                               res_count_color)

            if include_leaders:
                for leader in range(ANT_LEADERS_NUMBER):
                    position = ant_leader_agents[leader].position
                    scaled_position = (position[0] * x_scaled, position[1] * y_scaled)
                    ant_leader_images[leader].move(scaled_position)
                    screen.blit(ant_leader_images[leader].get_turned_image(),
                                (ant_leader_images[leader].get_position()[0] - ANT_LEADER_SIZE[0] // 2,
                                 ant_leader_images[leader].get_position()[1] - ANT_LEADER_SIZE[1] // 2))

        env.evaporation()
        pg.display.update()
        env.increment_env_tick()

        if not env.env_tick % 100:
            end = time.time()
            average_score = sum(mean_total_reward) / len(mean_total_reward)
            average_epsilon = sum(mean_epsilon) / len(mean_epsilon)
            print(f'Tick # {env.env_tick}, time: {round((end - start), 2)}')
            print(f'Epsilon = {round(average_epsilon, 5)}, Mean total reward = {average_score}')
            print(f'Unvisited q_table cells: {(q_table == 1).sum()}')
            print()
            start = time.time()

        # clock.tick(10)

    if learn_qt:
        with open("ants.pkl", 'wb') as f:
            pickle.dump(q_table, f)
    pg.quit()
