# ============================================================
# CS F425 Deep Learning — Phase 1
# agent.py  |  Data Analysis Agent
# ============================================================
# Built around the professor's ToolExecutor API:
#   executor.execute([{"tool": ..., "args": {...}}, ...])
#
# The 3 action patterns in the training data:
#   Pattern A (2-step):  filter_data → aggregate_sum
#   Pattern B (5-step):  filter_data → group_by → aggregate_sum → sort_by → top_k
#   Pattern C (2-step):  group_by → aggregate_mean
# ============================================================

import json
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------
# System prompt — grounded in the real tool signatures
# ----------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a data analysis agent. You have access to these tools that operate on a sales dataset.

The dataset has columns: date, year, month, city, region, product, category, revenue, units_sold, cost, profit

TOOLS (use EXACTLY these names and argument formats):
- filter_data(column='<col>', value=<int>)          — filter rows where column == value
- group_by(column='<col>')                           — group rows by a column
- aggregate_sum(column='<col>')                      — sum a column (after group_by for per-group)
- aggregate_mean(column='<col>')                     — mean of a column (after group_by for per-group)
- sort_by(column='<col>', order='desc')              — sort descending
- sort_by(column='<col>', order='asc')               — sort ascending
- top_k(k=<int>)                                     — return top k rows

PATTERNS — you must use ONE of these three sequences:
  Pattern A: filter_data → aggregate_sum
  Pattern B: filter_data → group_by → aggregate_sum → sort_by → top_k
  Pattern C: group_by → aggregate_mean

OUTPUT: Valid JSON only. No extra text.
{
  "actions": ["tool_name(arg='val', ...)", ...],
  "answer": <computed result or null>
}

EXAMPLES:

User: What is total revenue for 2022?
{"actions":["filter_data(column='year', value=2022)","aggregate_sum(column='revenue')"],"answer":null}

User: List top 3 cities based on revenue for 2021
{"actions":["filter_data(column='year', value=2021)","group_by(column='city')","aggregate_sum(column='revenue')","sort_by(column='revenue', order='desc')","top_k(k=3)"],"answer":null}

User: Average revenue by city
{"actions":["group_by(column='city')","aggregate_mean(column='revenue')"],"answer":null}

User: Top 5 regions by units_sold in 2022
{"actions":["filter_data(column='year', value=2022)","group_by(column='region')","aggregate_sum(column='units_sold')","sort_by(column='units_sold', order='desc')","top_k(k=5)"],"answer":null}

User: What is mean units_sold for each city?
{"actions":["group_by(column='city')","aggregate_mean(column='units_sold')"],"answer":null}

User: Total profit in 2023
{"actions":["filter_data(column='year', value=2023)","aggregate_sum(column='profit')"],"answer":null}
"""


# ----------------------------------------------------------------
# Model loader
# ----------------------------------------------------------------

_model     = None
_tokenizer = None


def load_model(model_id: str = MODEL_ID, adapter_path: str | None = None):
    global _model, _tokenizer
    print(f"[Agent] Loading tokenizer: {model_id}")
    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _tokenizer.pad_token = _tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    print("[Agent] Loading model (4-bit)...")
    _model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    if adapter_path:
        print(f"[Agent] Loading LoRA adapter: {adapter_path}")
        _model = PeftModel.from_pretrained(_model, adapter_path)
    _model.eval()
    print("[Agent] Ready.")


# ----------------------------------------------------------------
# Prompt builder
# ----------------------------------------------------------------

def build_prompt(query: str) -> str:
    return f"<s>[INST] {SYSTEM_PROMPT}\n\nUser: {query.strip()} [/INST]"


# ----------------------------------------------------------------
# JSON extractor — handles markdown fences, single quotes
# ----------------------------------------------------------------

def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in: {text[:200]}")
    depth = end = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i + 1; break
    raw = text[start:end]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)   # trailing commas
    raw = raw.replace("'", '"')                 # single → double quotes
    return json.loads(raw)


# ----------------------------------------------------------------
# Parse action strings → ToolExecutor dict format
# ----------------------------------------------------------------

def action_str_to_dict(action_str: str) -> dict:
    """
    Convert  "filter_data(column='year', value=2022)"
    →        {"tool": "filter", "args": {"column": "year", "op": "==", "value": 2022}}
    """
    s = action_str.strip()

    if s.startswith("filter_data"):
        m = re.search(r"column='([^']+)',\s*value=(\S+)\)", s)
        if m:
            val = m.group(2).strip("'\"")
            try: val = int(val)
            except ValueError:
                try: val = float(val)
                except ValueError: pass
            return {"tool": "filter", "args": {"column": m.group(1), "op": "==", "value": val}}

    elif s.startswith("group_by"):
        m = re.search(r"column='([^']+)'", s)
        if m:
            return {"tool": "groupby", "args": {"column": m.group(1)}}

    elif s.startswith("aggregate_sum"):
        m = re.search(r"column='([^']+)'", s)
        if m:
            return {"tool": "aggregate", "args": {"column": m.group(1), "agg": "sum"}}

    elif s.startswith("aggregate_mean"):
        m = re.search(r"column='([^']+)'", s)
        if m:
            return {"tool": "aggregate", "args": {"column": m.group(1), "agg": "mean"}}

    elif s.startswith("sort_by"):
        m = re.search(r"column='([^']+)',\s*order='([^']+)'", s)
        if m:
            return {"tool": "sort", "args": {"column": m.group(1), "ascending": m.group(2) == "asc"}}

    elif s.startswith("top_k"):
        m = re.search(r"k=(\d+)", s)
        if m:
            return {"tool": "topk", "args": {"k": int(m.group(1))}}

    raise ValueError(f"Cannot parse action: {s}")


# ----------------------------------------------------------------
# Core run function
# ----------------------------------------------------------------

def run_agent(query: str, df) -> dict:
    """
    Args:
        query : natural language question
        df    : pandas DataFrame (sales data)
    Returns:
        {"query": ..., "actions": [...], "answer": <DataFrame or scalar>}
    """
    if _model is None:
        raise RuntimeError("Call load_model() first.")

    from tool_executor import ToolExecutor

    prompt = build_prompt(query)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    raw = _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    agent_json = extract_json(raw)
    action_strings = agent_json["actions"]

    # Convert to ToolExecutor format and execute
    parsed = [action_str_to_dict(a) for a in action_strings]
    executor = ToolExecutor(df)
    result = executor.execute(parsed)

    return {
        "query":   query,
        "actions": action_strings,
        "answer":  result,
    }


# ----------------------------------------------------------------
# Batch evaluation
# ----------------------------------------------------------------

def evaluate(trajectories_path: str, df, output_path: str = "eval_results.txt"):
    import json
    with open(trajectories_path) as f:
        data = json.load(f)

    lines = []
    correct = 0
    for entry in data:
        query      = entry["query"]
        gold_actions = entry["actions"]
        try:
            result = run_agent(query, df)
            pred_actions = result["actions"]
            match = pred_actions == gold_actions
            if match: correct += 1
            lines.append(f"[{'OK' if match else 'X'}] {query}")
            if not match:
                lines.append(f"  Expected: {gold_actions}")
                lines.append(f"  Got:      {pred_actions}")
        except Exception as e:
            lines.append(f"[ERR] {query}: {e}")
        lines.append("")

    acc = correct / len(data)
    lines.insert(0, f"Accuracy: {acc:.1%} ({correct}/{len(data)})\n{'='*50}\n")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Accuracy: {acc:.1%} — saved to {output_path}")
    return acc


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

if __name__ == "__main__":
    import sys, pandas as pd
    csv_path     = sys.argv[1] if len(sys.argv) > 1 else "sales_data.csv"
    adapter_path = sys.argv[2] if len(sys.argv) > 2 else None
    df = pd.read_csv(csv_path)

    load_model(adapter_path=adapter_path)

    for q in [
        "What is total revenue for 2022?",
        "List top 3 cities based on revenue for 2021",
        "Average revenue by city",
    ]:
        r = run_agent(q, df)
        print(f"\nQ: {r['query']}")
        print(f"Actions: {r['actions']}")
        print(f"Answer:\n{r['answer']}")
