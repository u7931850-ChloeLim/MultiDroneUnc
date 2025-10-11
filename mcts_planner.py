import math
import random

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


def MCTS(initial_state, num_iterations=100):
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
def SelectChildUCB(parent_node, c_param=0.5):
    # Initialise
    best_child = None
    best_value = -float('inf')

    # Prevent log(0) when parent node's visit count is 0
    parent_visits = max(1, parent_node.visits)

    # Go through all children of the parent node
    for child in parent_node.children:
        # If the number of visits of the child node is 0, assign a huge value to encourage exploration
        if child.visits == 0:
            ucb_value = 1e6
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
    next_state, reward, done, info = env.simulate(node.state, action)

    # Create a new child node
    new_node = CreateNode(next_state, parent=node, action=action)

    # Add a new node to the list of children
    node.children.append(new_node)

    return new_node


# Choss an action that gets the drone closer to the goal without any collision
def HeuristicAction(state, num_candidates=26, samples=5):

    # Goal position of all drones
    cfg = env.get_config()
    goals = np.array(cfg.goal_positions)

    # All possible actions
    all_actions = list(range(env.num_actions))

    # Select candidate actions
    if (num_candidates is None) or (num_candidates >= len(all_actions)):
        candidates = all_actions
    else:
        candidates = random.sample(all_actions, num_candidates)

    # Calculate the current position and distance to goals
    cur_pos = state[:,:3]
    cur_dist = np.array([np.linalg.norm(cur_pos[i] - goals[i]) for i in range(len(goals))])

    # Set the default action
    best_action = candidates[0] if candidates else all_actions[0]
    # Set the initial value
    best_score = -float('inf')
    for action in candidates:
        score_sum = 0.0

        # Repeat simulate a few times to average out random effects
        for _ in range(samples):
            next_state, reward, done, info = env.simulate(state, action)

            # Drones will move within the grid
            grid_x, grid_y, grid_z = env.get_config().grid_size
            next_state[:, 0] =  np.clip(next_state[:, 0], 0, grid_x -1)
            next_state[:, 1] = np.clip(next_state[:, 1], 0, grid_y -1)
            next_state[:, 2] = np.clip(next_state[:, 2], 0, grid_z -1)

            # Calculate the distance between the current position and the goal
            next_pos = next_state[:, :3]
            next_dist = np.array([np.linalg.norm(next_pos[i] - goals[i]) for i in range(len(goals))])

            # Calculate how much each drone has moved closer to its goal
            delta = cur_dist - next_dist
            delta_pos = np.maximum(delta, 0.0)

            # Average progress and the slowest progress
            team_progress = np.mean(delta_pos)
            worst_progress = np.min(delta_pos)

            # Check if there is a collision
            num_collisions = info.get("num_collisions", 0)
            num_vehicle_collisions = info.get("num_vehicle_collisions", 0)

            # Apply a strong penalty if any collision occurs
            safety_penalty = 0.0
            if(num_collisions > 0) or (num_vehicle_collisions > 0):
                safety_penalty = -10000.0

            # Give a reward if the drones get closer to their goals
            proximity = np.sum(1.0 / (1.0 + next_dist)) * 10.0

            # Give a reward if a drone actually reaches its goal
            reach_bonus = 0.0
            for d in next_dist:
                if d < 0.5: # Inside the goal
                    reach_bonus += cfg.goal_reward * 5.0
                elif d < 1.5: # Close to goal
                    reach_bonus += cfg.goal_reward * 0.1
                else:
                    reach_bonus += 0.0

            # Detect if any drone has reached its goal and give a large reward
            goal_reached = [np.linalg.norm(next_pos[i] - goals[i]) < 0.5 for i in range(len(goals))]
            if any(goal_reached):
                sample_score = cfg.goal_reward * 10.0

            # otherwise, use the heuristic combination
            else:
                sample_score = (
                    20.0 * team_progress
                    + 10.0 * worst_progress
                    + proximity
                    + reach_bonus
                    + safety_penalty
                )
            score_sum += sample_score

        # Get the average over all samples for this action
        final_score = score_sum / samples

        # Update the best score and the best action
        if final_score > best_score:
            best_score = final_score
            best_action = action

    return best_action


# Run a random game from the current node’s state until the end
def Simulate(node):
    # Starts from the current node
    current_state = node.state
    total_reward = 0
    cfg = env.get_config()

    # Add a limit
    max_rollout = 5

    for _ in range(max_rollout):
        # Choose action with heuristic
        action = HeuristicAction(current_state, num_candidates=8, samples=3)

        # Simulate that action in the environment
        next_state, reward, done, info = env.simulate(current_state, action)

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
    def __init__(self, env, c_param = 0.5):
        self.env = env
        self.c_param = c_param

    def plan(self, current_state, planning_time_per_step=1.0):
        global env
        env = self.env
        return MCTS(current_state)