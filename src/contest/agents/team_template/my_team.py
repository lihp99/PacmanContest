# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent as CaptureAgent
from contest.game import Directions as Directions
from contest.util import nearest_point as nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OurAgent', second='OurAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}



def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

class AstarGen:
    def __init__(self, goal_locations, heuristic):
        self.goal_locations = set(goal_locations)
        self.heuristic = heuristic

    def get_next_action(self, problem):

        visited_nodes = set()
        frontier = util.PriorityQueue()
        start_state = ()

        # Push the initial state onto the frontier
        frontier.push((start_state, [], 0), self.heuristic(start_state, problem))

        while not frontier.is_empty():
            current_state, path, cost_sum = frontier.pop()

            if current_state in self.goal_locations:
                return path

            if current_state not in visited_nodes:
                visited_nodes.add(current_state)
                for successor, action, step_cost in problem.get_successors(current_state):
                    if successor not in visited_nodes:
                        # Calculate total cost and priority
                        new_cost = cost_sum + step_cost
                        priority = new_cost + self.heuristic(successor, problem)
                        frontier.update((successor, path + [action], new_cost), priority)

        return None


class MinimaxGen:
    def __init__(self, depth, evaluation_function, start_index):
        self.depth = depth
        self.evaluation_function = evaluation_function
        self.start_index = start_index

    def minimax(self, state, depth, agent_index, num_agents):
        """Perform the Minimax algorithm."""
        # Check if game is over or max depth reached


        if state.is_over() or depth == 0:
            return self.evaluation_function(state), None

        # Skip unobservable opponent agents
        while not state.get_agent_position(agent_index):
            print(f"Skipping unobservable agent: {agent_index}")
            agent_index = (agent_index + 1) % num_agents
            # Decrease depth only after a full cycle
            if agent_index == self.start_index:
                depth -= 1
            if depth == 0:
                return self.evaluation_function(state), None

        # Determine if this agent is maximizing or minimizing
        is_maximizing = (agent_index % 2 == 0) == state.is_on_red_team(self.start_index)  # Red: 0, 2; Blue: 1, 3

        print(f"Agent Index: {agent_index}, Depth: {depth}, Is Maximizing: {is_maximizing}")
        print(f"Agent {agent_index} Legal Actions: {state.get_legal_actions(agent_index)}")

        if is_maximizing:
            return self.max_value(state, depth, agent_index, num_agents)
        else:
            return self.min_value(state, depth, agent_index, num_agents)

    def max_value(self, state, depth, agent_index, num_agents):
        """Maximizing team's turn."""
        best_value = float('-inf')
        best_action = None
        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)
            value, _ = self.minimax(
                successor,
                depth - 1 if (agent_index + 1) % num_agents == self.start_index else depth,
                (agent_index + 1) % num_agents,
                num_agents
            )
            if value > best_value:
                best_value, best_action = value, action
        return best_value, best_action

    def min_value(self, state, depth, agent_index, num_agents):
        """Minimizing opponent's turn."""
        best_value = float('inf')
        best_action = None
        for action in state.get_legal_actions(agent_index):
            successor = state.generate_successor(agent_index, action)
            value, _ = self.minimax(
                successor,
                depth - 1 if (agent_index + 1) % num_agents == self.start_index else depth,
                (agent_index + 1) % num_agents,
                num_agents
            )
            if value < best_value:
                best_value, best_action = value, action
        return best_value, best_action

    def get_next_action(self, state):
        num_agents = len(state.get_red_team_indices() + state.get_blue_team_indices())
        #print(state.get_red_team_indices(), state.get_blue_team_indices())
        _, best_action = self.minimax(state, self.depth, self.start_index, num_agents)
        return best_action





class OurAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)


        self.modes = ['get_home', 'park_the_bus', 'sneaky_pellet']
        self.mode = 'park_the_bus'
        self.red = None

        #self.astar_gen_pellet = AstarGen(goal_type="pellet")
        #self.astar_gen_home = AstarGen(goal_type="home")
        self.minimax_gen = MinimaxGen(depth=3, evaluation_function=self.park_that_bus, start_index=self.index)


    def park_that_bus(self, game_state):
        print(self.red)
        desired_column = 6 if self.red else 7
        desired_row = 7
        agent_pos = game_state.get_agent_position(self.index)
        print(agent_pos)
        return -1000 * abs(game_state.get_agent_position(self.index)[1] - desired_row)    # -game_state.get_num_agents() * 1000 + self.index

    def choose_action(self, game_state):
        print(dir(game_state))
        self.red = game_state.is_on_red_team(self.index)
        if self.mode == 'sneaky_pellet':
            return self.astar_gen_pellet.get_next_action()
        elif self.mode == 'sneaky_pellet':
            return self.astar_gen_home.get_next_action()
        elif self.mode == 'park_the_bus':
            print(game_state.get_red_team_indices(), game_state.get_blue_team_indices())
            return self.minimax_gen.get_next_action(game_state)
        else:
            return
