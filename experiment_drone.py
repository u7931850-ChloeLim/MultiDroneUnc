# Disable vedo visualization completely
import vedo

# Dummy Plotter class to override vedo.plotter
class FakePlotter:
    def __init__(self, *args, **kwargs): pass
    def show(self, *args, **kwargs): pass
    def remove(self, *args, **kwargs): pass
    def render(self, *args, **kwargs): pass
    def add(self, *args, **kwargs): pass
    def interactive(self, *args, **kwargs): pass

# Dummy Axes class to override vedo.Axes
class FakeAxes:
    def __init__(self, *args, **kwargs): pass

# Disables visualisation
vedo.Plotter = FakePlotter
vedo.Axes = FakeAxes

import os
import numpy as np
import glob
import argparse
from scipy import stats
from multi_drone import MultiDroneUnc
from mcts_planner import MyPlanner


# Compute the mean and confidence interval for the values
def compute_confidence(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, margin

def run(env, planner, planning_time_per_step=1.0):
    current_state = env.reset()
    num_steps = 0
    total_discounted_reward = 0.0
    history = []

    while True:
        action = planner.plan(current_state, planning_time_per_step)
        next_state, reward, done, info = env.step(action)
        total_discounted_reward += (env.get_config().discount_factor ** num_steps) * reward
        history.append((current_state, action, reward, next_state, done, info))
        current_state = next_state
        num_steps += 1
        if done or num_steps >= env.get_config().max_num_steps:
            break
    return total_discounted_reward, history


# Main simulation runner
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, help="Path to the yaml configuration file")
    args = parser.parse_args()

    planning_time_per_step = 1.0
    num_runs = 20

    # If a specific config is given, use it
    if args.config:
        selected_files = [args.config]
    else:
        # Otherwise, auto-search for all drone_*.yaml files in the same folder
        base_dir = os.path.dirname(os.path.abspath(__file__))
        selected_files = sorted(glob.glob(os.path.join(base_dir, "drone_*.yaml")))

    if not selected_files:
        print("No YAML files found")
        return

    # Loop over each file
    for config_path in selected_files:
        print(f"\n Running simulation on: {os.path.basename(config_path)}")
        rewards = []
        successes = []

        # Run the simulation multiple times
        for i in range(num_runs):
            env = MultiDroneUnc(config_path)
            planner = MyPlanner(env, c_param=0.8)
            total_discounted_reward, history = run(env, planner, planning_time_per_step)

            rewards.append(total_discounted_reward)

            # Get success flag from the last entry in history
            success_flag = history[-1][5].get("success", False)
            successes.append(success_flag)

            print(f"Run {i+1}/{num_runs} - Reward: {total_discounted_reward:.2f}, Success: {success_flag}")

        # Compute statistics for reward and success rate
        mean_reward, ci_reward = compute_confidence(rewards)
        success_rate = 100.0 * np.mean(successes)
        _, ci_success = compute_confidence(successes)

        # Print summary
        print(f"\n Summary for {os.path.basename(config_path)}:")
        print(f" → Mean Reward: {mean_reward:.2f} ± {ci_reward:.2f} (95% CI)")
        print(f" → Success Rate: {success_rate:.1f}% ± {ci_success * 100:.1f}% (95% CI)")


if __name__ == "__main__":
    main()
