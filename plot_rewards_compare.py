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
from datetime import datetime


def _load_entries(json_file_path):
    """Load reward entries from file, supporting two formats:
    - List[List[float]]
    - Dict with key 'entries' -> List[List[float]]
    Returns list of lists or None on error.
    """
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file_path}!")
        return None

    if (
        isinstance(data, dict)
        and "entries" in data
        and isinstance(data["entries"], list)
    ):
        return data["entries"]
    if isinstance(data, list):
        return data
    print(f"Error: Unsupported JSON structure in {json_file_path}")
    return None


def _avg_rewards_from_entries(entries):
    """Compute average reward per step given entries (list of lists)."""
    avg_rewards = []
    for step_rewards in entries:
        if isinstance(step_rewards, list) and len(step_rewards) > 0:
            try:
                rewards = [float(r) for r in step_rewards]
            except Exception:
                # Skip malformed rows
                continue
            avg_rewards.append(float(np.mean(rewards)))
    return avg_rewards


def plot_overlays(json_files, window_size=50, show_raw=False):
    """Overlay smoothed average reward curves from multiple JSON files.

    - json_files: list of file paths
    - window_size: smoothing window for moving average per series
    - show_raw: if True, plot raw per-step averages with low alpha
    """
    if not json_files:
        print("No files provided to plot.")
        return

    plt.figure(figsize=(12, 8))

    any_series = False
    stats_lines = []

    # Distinct color cycle per file
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(json_files))))

    for idx, json_path in enumerate(json_files):
        entries = _load_entries(json_path)
        if not entries:
            continue
        avg_rewards = _avg_rewards_from_entries(entries)
        if not avg_rewards:
            print(f"Warning: No valid reward data in {json_path}")
            continue
        any_series = True

        # Optional raw line
        if show_raw:
            steps = range(len(avg_rewards))
            plt.plot(
                steps,
                avg_rewards,
                color=colors[idx % len(colors)],
                linewidth=1,
                alpha=0.25,
                label=f"Raw: {Path(json_path).stem}",
            )

        # Smoothed line
        actual_window = max(1, min(int(window_size), max(1, len(avg_rewards) // 2)))
        if actual_window > 1:
            moving_avg = np.convolve(
                avg_rewards, np.ones(actual_window) / actual_window, mode="valid"
            )
            # Centered x positions for the valid convolution
            offset = actual_window // 2
            moving_steps = range(offset, offset + len(moving_avg))
            label = f"{Path(json_path).stem} (w={actual_window})"
            plt.plot(
                moving_steps,
                moving_avg,
                color=colors[idx % len(colors)],
                linewidth=2.5,
                alpha=0.9,
                label=label,
            )
        else:
            steps = range(len(avg_rewards))
            plt.plot(
                steps,
                avg_rewards,
                color=colors[idx % len(colors)],
                linewidth=2.5,
                alpha=0.9,
                label=f"{Path(json_path).stem}",
            )

        # Series stats
        stats_lines.append(
            f"{Path(json_path).stem}: mean={np.mean(avg_rewards):.3f}, max={np.max(avg_rewards):.3f}, final={avg_rewards[-1]:.3f}"
        )

    if not any_series:
        print("No plottable series found.")
        return

    # Formatting
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("GRPO Training: Smoothed Average Reward per Step", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.05)

    # Stats footer
    if stats_lines:
        plt.figtext(
            0.5,
            0.02,
            " | ".join(stats_lines),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightgray"),
        )

    plt.tight_layout()

    # Save combined figure near the first file
    first = Path(json_files[0])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = first.parent / f"overlay_rewards_{ts}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Overlay plot saved to: {out_path}")

    plt.show()

    return out_path


def main():
    # Default files (add more paths to compare by default)
    default_files = [
        "/home/zy1130/LifelongAgentBench_env_reward/grpo_logs/Qwen_Qwen2.5-Coder-7B-Instruct_20250831_210814/reward_history_20250831_210814.json",
        "/home/zy1130/LifelongAgentBench_env_reward/grpo_logs/Qwen_Qwen2.5-Coder-7B-Instruct_20250901_120401/reward_history_20250901_120401.json",
        "/home/zy1130/LifelongAgentBench_env_reward/grpo_logs/Qwen_Qwen2.5-Coder-7B-Instruct_20250901_212215/reward_history_20250901_212215.json",
        "/home/zy1130/LifelongAgentBench_env_reward/grpo_logs/Qwen_Qwen2.5-Coder-7B-Instruct_20250902_131118/reward_history_20250902_131118.json",
        "/home/zy1130/LifelongAgentBench_env_reward/grpo_logs/Qwen_Qwen2.5-Coder-7B-Instruct_20250903_012914/reward_history_20250903_012914.json",
    ]
    # CLI usage patterns supported:
    #   python plot_rewards copy.py                -> uses default_files with window_size=50
    #   python plot_rewards copy.py 100            -> uses default_files with window_size=100
    #   python plot_rewards copy.py file1.json     -> window_size=50, files=[file1]
    #   python plot_rewards copy.py file1 file2    -> window_size=50, files=[file1, file2]
    #   python plot_rewards copy.py file1 75       -> window_size=75, files=[file1]
    #   python plot_rewards copy.py file1 file2 75 -> window_size=75, files=[file1, file2]

    args = sys.argv[1:]
    window_size = 50
    files = []

    if not args:
        files = default_files
    else:
        # Heuristic: last arg numeric -> window size
        last = args[-1]
        if last.isdigit():
            try:
                ws = int(last)
                if ws >= 1:
                    window_size = ws
                    args = args[:-1]
            except Exception:
                pass
        # Remaining args considered file paths; if none remain, use defaults
        files = args if args else default_files

    # Expand user and validate paths
    files = [str(Path(p).expanduser()) for p in files]

    print("Overlaying rewards from:")
    for p in files:
        print(f" - {p}")
    print(f"Moving average window size: {window_size}")

    plot_overlays(files, window_size=window_size, show_raw=False)


if __name__ == "__main__":
    main()
