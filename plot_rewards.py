#!/usr/bin/env python3
"""
Plot reward history from GRPO training logs.
Computes the average of each group of 4 candidate rewards per training step.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path


def plot_reward_history(json_file_path, window_size=10):
    """Plot the average rewards over training steps."""

    # Load the JSON data
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found!")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file_path}!")
        return

    if not data:
        print("Error: Empty data!")
        return

    # Compute average reward for each step
    avg_rewards = []
    for step_rewards in data:
        if isinstance(step_rewards, list) and len(step_rewards) > 0:
            # Convert to float and compute average
            rewards = [float(r) for r in step_rewards]
            avg_reward = np.mean(rewards)
            avg_rewards.append(avg_reward)
        else:
            print(f"Warning: Skipping invalid step data: {step_rewards}")

    if not avg_rewards:
        print("Error: No valid reward data found!")
        return

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot average rewards
    steps = range(len(avg_rewards))
    plt.plot(steps, avg_rewards, "b-", linewidth=2, alpha=0.8, label="Average Reward")

    # Add moving average for smoothing
    if len(avg_rewards) >= window_size:
        # Ensure window size doesn't exceed half the data length
        actual_window = min(window_size, len(avg_rewards) // 2)
        if actual_window > 1:
            moving_avg = np.convolve(
                avg_rewards, np.ones(actual_window) / actual_window, mode="valid"
            )
            moving_steps = range(
                actual_window // 2, len(avg_rewards) - actual_window // 2 + 1
            )
            plt.plot(
                moving_steps,
                moving_avg,
                "r-",
                linewidth=3,
                alpha=0.7,
                label=f"Moving Average (window={actual_window})",
            )

    # Formatting
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(
        "GRPO Training: Average Reward per Step\n(Average of 4 candidate rewards per step)",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set y-axis limits to show the full [0, 1] range
    plt.ylim(-0.05, 1.05)

    # Add summary statistics
    mean_reward = np.mean(avg_rewards)
    max_reward = np.max(avg_rewards)
    min_reward = np.min(avg_rewards)
    final_reward = avg_rewards[-1]

    stats_text = f"Mean: {mean_reward:.3f} | Max: {max_reward:.3f} | Min: {min_reward:.3f} | Final: {final_reward:.3f}"
    plt.figtext(
        0.5,
        0.02,
        stats_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgray"),
    )

    plt.tight_layout()

    # Save the plot
    output_file = Path(json_file_path).with_suffix(".png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")

    # Show the plot
    plt.show()

    # Print summary
    print(f"\nSummary Statistics:")
    print(f"Total training steps: {len(avg_rewards)}")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Max reward: {max_reward:.3f}")
    print(f"Min reward: {min_reward:.3f}")
    print(f"Final reward: {final_reward:.3f}")

    return avg_rewards


def main():
    # Default file path
    default_file = "/home/zy1130/LifelongAgentBench_env_reward/grpo_logs/Qwen_Qwen2.5-Coder-7B-Instruct_20250903_012914/reward_history_20250903_012914.json"

    # Parse command line arguments
    json_file = default_file
    window_size = 50  # Default window size

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            window_size = int(sys.argv[2])
            if window_size < 1:
                print("Error: Window size must be at least 1")
                return
        except ValueError:
            print("Error: Window size must be an integer")
            print("Usage: python plot_rewards.py [json_file] [window_size]")
            return

    print(f"Plotting rewards from: {json_file}")
    print(f"Moving average window size: {window_size}")
    plot_reward_history(json_file, window_size)


if __name__ == "__main__":
    main()
