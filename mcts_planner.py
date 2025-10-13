import math
import random
import time

import numpy as np

env = None

# Define the Node class to represent each state in the MCTS tree
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0 # the number of times this node was visited
        self.total_reward = 0 # total rewards accumulated through this node
        self.untried_actions = [] # actions that haven't been tried yet


# Create a new node and return it
def CreateNode(state, parent=None, action=None):
    node = Node(state, parent, action)
    # Get all possible actions from this state
    node.untried_actions = Get_Available_Actions(state)
    return node


# Get all possible actions from the state
def Get_Available_Actions(state):
    return list(range(env.num_actions))


def MCTS(initial_state, num_iterations=20):
    # Create the root node
    root_node = CreateNode(initial_state)

    # Repeat for a fixed number of iterations
    for i in range(num_iterations):

        # 1.Select the node to explore further
        selected_node = SelectNode(root_node)

        # 2.Expand one of its unexplored children
        expanded_node = Expand(selected_node)

        # 3.Rollout: Simulate a full game from expanded_node
        reward = Simulate(expanded_node)

        # 4.Backpropagation: Update the value of the node visited in this iteration
        Backpropagate(expanded_node, reward)

    # Return the action that leads to the best child of the root
    best_action = BestAction(root_node)
    return best_action


# Select a node in the tree to explore further
def SelectNode(node):
    # Go down while node is not a leaf node AND node is fully expanded
    while len(node.untried_actions) == 0 and len(node.children) > 0:
        node = SelectChildUCB(node)
    return node


# Select a child node using UCB
def SelectChildUCB(parent_node, c_param=0.8):
    # Initialise
    best_child = None
    best_value = -float('inf')

    # Prevent log(0) when parent node's visit count is 0
    parent_visits = max(1, parent_node.visits)

    # Go through all children of the parent node
    for child in parent_node.children:
        # If the number of visits of the child node is 0, assign a huge value to encourage exploration
        if child.visits == 0:
            ucb_value = 1000 + c_param * math.sqrt(math.log(parent_visits) / child.visits)
        else:
            # UCB formula: argmaxQ(s,a) + c*sqrt(ln(n(s))/n(s,a))
            exploitation = child.total_reward / child.visits
            exploration = c_param * math.sqrt(math.log(parent_visits) / (child.visits))
            # Combine both of them
            ucb_value = exploitation + exploration

        # Update best value and best child
        if ucb_value > best_value:
            best_value = ucb_value
            best_child = child

    # If all UCB values are equal, pick a child node randomly
    if best_child is None:
        best_child = random.choice(parent_node.children)

    return best_child


# Expand the selected node by trying one untried action
def Expand(node):
    # If all actions were tried already, return the node itself
    if len(node.untried_actions) == 0:
        return node

    # Pick one action randomly
    index = random.randrange(len(node.untried_actions))
    action = node.untried_actions.pop(index)

    # Simulate it in the environment
    next_state, _, _, _ = env.simulate(node.state, action)

    # Create a new child node
    new_node = CreateNode(next_state, parent=node, action=action)

    # Add a new node to the list of children
    node.children.append(new_node)

    return new_node


# Choose an action that gets the drone closer to the goal without any collision
def HeuristicAction(state, samples):

    # Goal position of all drones
    cfg = env.get_config()
    goals = np.array(cfg.goal_positions)

    # All possible actions
    all_actions = list(range(env.num_actions))

    obstacles = np.array(cfg.obstacle_cells)

    # Calculate the current position and distance to goals
    cur_pos = state[:,:3]
    cur_dist = np.linalg.norm(cur_pos - goals, axis=1)

    # Set the default action
    best_action = all_actions[0]
    # Set the initial value
    best_score = -float('inf')

    for action in all_actions:
        score_sum = 0.0

        # Repeat simulate a few times to average out random effects
        for _ in range(samples):
            next_state, _, _, info = env.simulate(state, action)

            # Drones will move within the grid
            grid_x, grid_y, grid_z = env.get_config().grid_size
            next_state[:, 0] =  np.clip(next_state[:, 0], 0, grid_x -1)
            next_state[:, 1] = np.clip(next_state[:, 1], 0, grid_y -1)
            next_state[:, 2] = np.clip(next_state[:, 2], 0, grid_z -1)

            # Calculate the distance between the current position and the goal
            next_pos = next_state[:, :3]
            next_dist = np.linalg.norm(next_pos - goals, axis=1)

            # Calculate how much each drone has moved closer to its goal
            delta = cur_dist - next_dist
            progress = np.mean(np.clip(delta, 0, None))

            # Give a reward when the drone gets closer to the goal
            near_goal_bonus = np.sum(next_dist < 1.0) * 25.0
            reach_bonus = np.sum(next_dist < 0.1) * 2000

            direction_to_goal = goals - cur_pos
            next_direction = next_pos - cur_pos
            alignment = np.mean(
                np.sum(direction_to_goal * next_direction, axis=1)
                / (np.linalg.norm(direction_to_goal, axis=1) * np.linalg.norm(next_direction, axis=1) + 1e-6)
            )
            alignment_bonus = 30.0 * alignment if np.mean(cur_dist) > 1.5 else 0.0


            # Calculate the collision penalty
            num_coll = info.get("num_collisions", 0)
            num_vcoll = info.get("num_vehicle_collisions", 0)
            collision_penalty = -30000 * (num_coll + num_vcoll)

            # Gives a penalty when the drone gets close to an obstacles
            near_obstacle_penalty = 0
            for drone_pos in next_pos:
                dist = np.linalg.norm(obstacles - drone_pos, axis=1)
                min_dist = np.min(dist)
                if min_dist < 1.0:
                    near_obstacle_penalty -= 15000 * (1.0 - min_dist)
                elif min_dist < 2.0:
                    near_obstacle_penalty -= 4000 * (2.0 - min_dist)

            # Gives a penalty when drones get close to each other
            drone_collision_penalty = 0
            num_drones = len(next_pos)
            for i in range(num_drones):
                for j in range(i + 1, num_drones):
                    dist = np.linalg.norm(next_pos[i] - next_pos[j])
                    if dist < 1.0:
                        drone_collision_penalty -= 20000 * (1.0 - dist)
                    elif dist < 2.0:
                        drone_collision_penalty -= 6000 * (2.0 - dist)


            # otherwise, use the heuristic combination
            sample_score = (
                20.0 * progress
                + alignment_bonus
                + near_goal_bonus
                + reach_bonus
                + collision_penalty
                + near_obstacle_penalty
                + drone_collision_penalty
                )
            score_sum += sample_score

        # Get the average over all samples for this action
        avg_score = score_sum / samples

        # Update the best score and the best action
        if avg_score > best_score:
            best_score = avg_score
            best_action = action

    return best_action


# Run a random game from the current node’s state until the end
def Simulate(node):
    # Starts from the current node
    current_state = node.state
    total_reward = 0

    # Add a limit
    max_rollout = 10

    for _ in range(max_rollout):
        # Choose action with heuristic
        action = HeuristicAction(current_state, samples=2)

        # Simulate that action in the environment
        next_state, reward, done, info = env.simulate(current_state, action)

        # Stop when collision occurs
        if info.get("num_collisions") or info.get("num_vehicle_collisions"):
            total_reward = -5000
            break

        total_reward += reward
        current_state = next_state

        # If goal reached or done, stop early
        if done:
            break

    return total_reward


# Update the value of the nodes visited in this iteration
def Backpropagate(node, reward):

    while node is not None:
        # Update the number of visits
        node.visits += 1

        # Accumulate the reward
        node.total_reward += reward

        # Go up to the parent node
        node = node.parent


# Get the best action that led to the best child
def BestAction(root_node):
    # Initialise
    best_action = None
    max_avg_reward = -float('inf')

    # Go through all child nodes of the root
    for child in root_node.children:
        if child.visits > 0:
            # Calculate the average of the total reward
            avg_reward = child.total_reward / child.visits
            if avg_reward > max_avg_reward:
                max_avg_reward = avg_reward
                best_action = child.action
    return best_action


class MyPlanner:
    def __init__(self, env, c_param = 0.8):
        self.env = env
        self.c_param = c_param

    def plan(self, current_state, planning_time_per_step:float):
        global env
        env = self.env

        start_time = time.time()
        best_action = None
        best_value = -float('inf')

        while time.time() - start_time < planning_time_per_step:
            action = MCTS(current_state)
            next_state, reward, done, info = env.simulate(current_state, action)

            if reward > best_value:
                best_value = reward
                best_action = action

        return best_action