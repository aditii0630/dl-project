import pandas as pd
import json
import re
from tool_executor import ToolExecutor

def parse_agent_action(action_str):
    """
    Parses a string-based agent action into the dictionary format
    required by the ToolExecutor.
    """
    if action_str.startswith("filter_data"):
        # Extracts column name and numeric value
        match = re.search(r"column='([^']+)', value=(\d+)", action_str)
        if match:
            return {
                "tool": "filter", 
                "args": {"column": match.group(1), "op": "==", "value": int(match.group(2))}
            }

    elif action_str.startswith("group_by"):
        match = re.search(r"column='([^']+)'", action_str)
        if match:
            return {"tool": "groupby", "args": {"column": match.group(1)}}

    elif action_str.startswith("aggregate_sum"):
        match = re.search(r"column='([^']+)'", action_str)
        if match:
            return {"tool": "aggregate", "args": {"column": match.group(1), "agg": "sum"}}

    # FIX: aggregate_mean was missing from the original
    elif action_str.startswith("aggregate_mean"):
        match = re.search(r"column='([^']+)'", action_str)
        if match:
            return {"tool": "aggregate", "args": {"column": match.group(1), "agg": "mean"}}

    elif action_str.startswith("sort_by"):
        match = re.search(r"column='([^']+)', order='([^']+)'", action_str)
        if match:
            # Convert 'desc' to ascending=False, otherwise True
            is_ascending = False if match.group(2) == 'desc' else True
            return {"tool": "sort", "args": {"column": match.group(1), "ascending": is_ascending}}

    elif action_str.startswith("top_k"):
        match = re.search(r"k=(\d+)", action_str)
        if match:
            return {"tool": "topk", "args": {"k": int(match.group(1))}}

    raise ValueError(f"Could not parse action: {action_str}")

def run_automated_pipeline(csv_path, json_path, output_path="pipeline_results.txt"):
    # 1. Load the data
    df = pd.read_csv(csv_path)

    # 2. Load the trajectories (queries and string actions)
    with open(json_path, 'r') as file:
        trajectories = json.load(file)

    lines = []

    # 3. Iterate through each query and its associated actions
    for entry in trajectories:
        query = entry["query"]
        raw_actions = entry["actions"]

        lines.append(f"Query: {query}")
        lines.append("-" * 40)

        # 4. Parse strings into dictionary format
        parsed_actions = []
        for action_str in raw_actions:
            parsed_action = parse_agent_action(action_str)
            parsed_actions.append(parsed_action)

        # 5. Initialize the executor and run
        executor = ToolExecutor(df)
        try:
            result_df = executor.execute(parsed_actions)
            lines.append("Answer:")
            lines.append(result_df.to_string(index=False))
        except Exception as e:
            lines.append(f"Error: {e}")

        lines.append("")

    # 6. Write all results to a text file
    with open(output_path, 'w') as out_file:
        out_file.write("\n".join(lines))

    print(f"Results written to {output_path}")

# Trigger the automated run
if __name__ == "__main__":
    run_automated_pipeline("sample_sales.csv", "sample_trajectories.json")