#!/usr/bin/env python3
"""
Script to download LifelongAgentBench data from Hugging Face and save it locally
in the format expected by the server configuration.
"""

import json
import os
from datasets import load_dataset
from pathlib import Path


def download_and_save_data():
    print("Downloading LifelongAgentBench data from Hugging Face...")

    # Download the dataset using built-in dataset config (Option A from feedback)
    ds = load_dataset(
        "csyq/LifelongAgentBench", data_dir="db_bench", split="train", streaming=True
    )

    print("Examining data structure...")

    # Check the format of the data (first few items)
    converted_data = {}

    for i, item in enumerate(ds.take(10)):  # Only take first 10 items to examine
        print(f"Item {i}:")
        print(f"  Keys: {list(item.keys())}")
        print(f"  Sample data: {item}")
        print("-" * 50)

        # Convert each item to the expected JSON format
        converted_data[str(i)] = item

        if i >= 2:  # Only show first 3 items in detail
            break

    # Now let's get more data for saving
    print("Loading full dataset for saving...")
    ds_full = load_dataset(
        "csyq/LifelongAgentBench",
        data_dir="db_bench",
        split="train",
        streaming=False,  # Now get all data
    )

    print(f"Full dataset has {len(ds_full)} samples")

    # Convert the full dataset
    converted_data = {}
    for i, item in enumerate(ds_full):
        converted_data[str(i)] = item
        if i >= 500:  # Limit to first 500 items to match the expected filename
            break

    # Ensure the output directory exists
    output_file = "data/v0303/db_bench/processed/v0317_first500/entry_dict.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the converted data
    with open(output_file, "w") as f:
        json.dump(converted_data, f, indent=2)

    print(f"Saved {len(converted_data)} items to {output_file}")


if __name__ == "__main__":
    download_and_save_data()
