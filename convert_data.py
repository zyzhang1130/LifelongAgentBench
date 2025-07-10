#!/usr/bin/env python3
"""
Convert the downloaded Hugging Face data to the format expected by DBBench.
The HF data has string representations of Python objects that need to be parsed.
"""

import json
import ast


def convert_hf_to_dbbench_format():
    print("Converting Hugging Face data to DBBench format...")

    # Load the downloaded data
    input_file = "data/v0303/db_bench/processed/v0317_first500/entry_dict.json"
    with open(input_file, "r") as f:
        hf_data = json.load(f)

    print(f"Loaded {len(hf_data)} items from Hugging Face data")

    converted_data = {}

    for key, item in hf_data.items():
        try:
            # Parse the string representations to Python objects
            answer_info = ast.literal_eval(item["answer_info"])
            table_info = ast.literal_eval(item["table_info"])
            skill_list = ast.literal_eval(item["skill_list"])

            # Convert to the format expected by DBBench
            converted_item = {
                "instruction": item["instruction"],
                "answer_info": {
                    "md5": answer_info.get("md5"),
                    "direct": answer_info.get("direct"),
                    "sql": answer_info.get(
                        "sql"
                    ),  # This should be 'ground_truth_sql' in the AnswerInfo model
                },
                "table_info": {
                    "name": table_info["name"],
                    "row_list": table_info["row_list"],
                    "column_info_list": table_info["column_info_list"],
                },
                "skill_list": skill_list,
            }

            converted_data[key] = converted_item

            # Show first item as example
            if key == "0":
                print("Example converted item:")
                print(json.dumps(converted_item, indent=2))

        except Exception as e:
            print(f"Error converting item {key}: {e}")
            continue

    # Save the converted data
    output_file = (
        "data/v0303/db_bench/processed/v0317_first500/entry_dict_converted.json"
    )
    with open(output_file, "w") as f:
        json.dump(converted_data, f, indent=2)

    print(f"Converted and saved {len(converted_data)} items to {output_file}")

    # Now replace the original file
    import shutil

    shutil.move(output_file, input_file)
    print(f"Replaced original file: {input_file}")


if __name__ == "__main__":
    convert_hf_to_dbbench_format()
