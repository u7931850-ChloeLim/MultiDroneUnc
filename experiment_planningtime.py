# Disable vedo visualization completely
import vedo

class FakePlotter:
    def __init__(self, *args, **kwargs): pass
    def show(self, *args, **kwargs): pass
    def remove(self, *args, **kwargs): pass
    def render(self, *args, **kwargs): pass
    def add(self, *args, **kwargs): pass
    def interactive(self, *args, **kwargs): pass

class FakeAxes:
    def __init__(self, *args, **kwargs): pass

vedo.Plotter = FakePlotter
vedo.Axes = FakeAxes  #  disables 3D visualization safely

# -----------------------------
# Imports
# -----------------------------
import os
import numpy as np
from scipy import stats
from multi_drone import MultiDroneUnc
from mcts_planner import MyPlanner
from run_u7931850 import run  # reuse your existing run() function

print(" experiment.py started")

# -----------------------------
# Helper: Compute confidence interval
# -----------------------------
def compute_confidence(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, margin

# -----------------------------
# Main
# -----------------------------
def main():
    config_dir = "C:/Users/ABC/PycharmProjects/MultiDroneUnc"
    num_runs = 20

    # Choose the YAML file to test
    config_name = "drone_2.yaml"
    config_path = os.path.join(config_dir, config_name)

    if not os.path.exists(config_path):
        print(f" File not found: {config_path}")
        return

    # Planning times to test
    planning_times = [0.5, 1.0, 2.0, 3.0, 4.0]

    print(f"===============================")
    print(f"  Running {config_name}")
    print(f"===============================")

    for planning_time in planning_times:
        print(f"\n Planning time per step: {planning_time:.1f}s")
        rewards = []

        for i in range(num_runs):
            env = MultiDroneUnc(config_path)
            planner = MyPlanner(env, c_param=0.5)
            total_reward, history = run(env, planner, planning_time_per_step=planning_time)
            rewards.append(total_reward)

            success = history[-1][5].get("success", None)
            print(f"Run {i+1:02d}/{num_runs}: Reward = {total_reward:.2f}, Success = {success}")

        mean, margin = compute_confidence(rewards)
        print(f"✅ Result ({planning_time:.1f}s): Mean reward = {mean:.2f} ± {margin:.2f} (95% CI)")

if __name__ == "__main__":
    main()
