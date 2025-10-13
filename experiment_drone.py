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

vedo.Plotter = FakePlotter
vedo.Axes = FakeAxes  # ✅ 추가!

import os
import numpy as np
from scipy import stats
from multi_drone import MultiDroneUnc
from mcts_planner import MyPlanner
from run_u7931850 import run  # run(env, planner, planning_time_per_step) 함수 재사용


# Compute the mean and confidence interval for the values
def compute_confidence(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, margin


# Main simulation runner
def main():

    planning_time_per_step = 1.0
    num_runs = 20

    selected_files = [
        "C:/Users/ABC/PycharmProjects/MultiDroneUnc/drone_1.yaml",
        "C:/Users/ABC/PycharmProjects/MultiDroneUnc/drone_2.yaml",
        "C:/Users/ABC/PycharmProjects/MultiDroneUnc/drone_3.yaml",
        "C:/Users/ABC/PycharmProjects/MultiDroneUnc/drone_4.yaml",
        "C:/Users/ABC/PycharmProjects/MultiDroneUnc/drone_5.yaml",
    ]

    # Loop over each file
    for config_path in selected_files:
        if not os.path.exists(config_path):
            print(f" {config_path} not found!")
            continue

        print(f"\n Running simulation on: {config_path}")
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
