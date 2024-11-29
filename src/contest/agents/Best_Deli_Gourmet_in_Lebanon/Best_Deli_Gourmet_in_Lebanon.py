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
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################



def create_team(first_index, second_index, is_red,
                first='DefensiveReflexAgent', second='AstarAgentOffensive', num_training=0):

# def create_team(first_index, second_index, is_red,
#                 first='OffensiveReflexAgent', second='AStarAgent', num_training=0):
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
        print(f"start:{self.start}")
        print("REGISTER_INITIAL_STATE")
        print(game_state.get_agent_state(self.index).get_position())
        CaptureAgent.register_initial_state(self, game_state)


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # print(f"\nREFLEX AGENT: game_state:\n{game_state}\n")
        # print(dir(game_state))
        # print("\n")
        actions = game_state.get_legal_actions(self.index)
        # print(f"\nREFLEX AGENT: type of agent:{self.index}")
        # print(f"agent.red:{self.red}")
        # print(f"\nREFLEX AGENT: actions:{actions}")

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

        # print(f"actions:{actions}")
        # print(f"best_actions:{best_actions}")
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
    




class OffensiveAstarAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def astar_search(self, initial_state, initial_pos, goal):
        frontier = util.PriorityQueue()
        frontier.push((initial_state, initial_pos, []), 0)
        explored = set()
        state_costs = {initial_pos: 0}  # Dictionary to track the minimum cost to reach each state


        while not frontier.is_empty():
            current_state, current_pos, path = frontier.pop()

            if current_pos == goal:
                return path

            if current_pos in explored:
                continue # skip expanded nodes

            else: explored.add(current_pos)

            for action in current_state.get_legal_actions(self.index):
                if action == Directions.STOP:
                    continue
                
                successor = current_state.generate_successor(self.index, action) #Returns the successor state (a GameState object) after the specified agent takes the action.
                next_pos = successor.get_agent_state(self.index).get_position() #Returns the position of agent after taken the current action
                next_pos = nearest_point(next_pos)

                if next_pos in explored:
                    continue

                new_cost = state_costs[current_pos] + 1
                
                # if the next position has not been explored or this is the cheapest path to the next node 
                if next_pos not in explored or new_cost < state_costs[next_pos]:
                    state_costs[next_pos] = new_cost
                    path_new = path + [action]

                    # f_node is based on f(n) = g(n) + h(n) with h(n) the heuristic and g(n) the cost to reach the current node
                    f_node = new_cost + self.get_maze_distance(next_pos, goal)
                    frontier.push((successor, next_pos, path_new), f_node)

    def choose_action(self, game_state):
            """
            Choose an action using A* to navigate to the nearest food.
            """
            my_pos = game_state.get_agent_state(self.index).get_position()
            food_list = self.get_food(game_state).as_list()
            # If there is no food left, stop
            if not food_list:
                return Directions.STOP
            
            goal = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
            path = self.astar_search(game_state, my_pos, goal)

            # here make sure that the dude goes back to base if he has less than 2 food left (so if path len < 2)
            return path[0] if path else Directions.STOP