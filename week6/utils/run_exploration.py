from pathlib import Path

from exploration import run_exploration

script_parent = Path(__file__).parent
root_output_dir = script_parent / "../data"

if __name__ == "__main__":
    n_trials = 3
    running_time = 20

    for seed in range(n_trials):
        trial_output_dir = root_output_dir / f"seed={seed}"
        run_exploration(seed, running_time, trial_output_dir)

    print("All tasks completed.")