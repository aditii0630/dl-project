# ============================================================
# CS F425 Deep Learning — Phase 2B
# react_agent.py  |  ReAct Execution Loop (Pure Python)
# ============================================================
# Calls professor's ToolExecutor under the hood.
# NO LangChain / LlamaIndex / AutoGen.
# ============================================================

import re
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
import pandas as pd
from tool_executor import ToolExecutor


# ================================================================
# 1. Stopping Criteria
# ================================================================

class ObservationStoppingCriteria(StoppingCriteria):
    """Halt generation the moment 'Observation:' is produced."""
    def __init__(self, tokenizer):
        self.stop_ids = tokenizer.encode("Observation:", add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0].tolist()[-len(self.stop_ids):] == self.stop_ids


class FinalAnswerStoppingCriteria(StoppingCriteria):
    """Halt generation when 'Final Answer:' is produced."""
    def __init__(self, tokenizer):
        self.stop_ids = tokenizer.encode("Final Answer:", add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0].tolist()[-len(self.stop_ids):] == self.stop_ids


# ================================================================
# 2. Robust Action Parser
# ================================================================

def parse_action_and_input(text: str):
    """
    Extract (action_name, action_input_dict) from a generated block.
    Handles: trailing commas, single quotes, markdown fences.
    Returns None if no Action found.
    """
    name_m = re.search(r"Action:\s*(\w+)", text)
    if not name_m:
        return None
    action_name = name_m.group(1).strip()

    input_m = re.search(r"Action Input:\s*(.+?)(?=\nThought:|\nObservation:|\nFinal Answer:|$)", text, re.DOTALL)
    if not input_m:
        return None
    raw = input_m.group(1).strip()

    # Sanitise
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    raw = raw[start:end]
    raw = re.sub(r"(?<![\\])'", '"', raw)        # single → double quotes
    raw = re.sub(r",\s*([}\]])", r"\1", raw)      # trailing commas
    raw = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', raw)  # unquoted keys

    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return action_name, args


def extract_final_answer(text: str):
    m = re.search(r"Final Answer:\s*(.+?)$", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ================================================================
# 3. Action dispatcher — maps Action name → ToolExecutor dict
# ================================================================

def dispatch(action_name: str, args: dict) -> dict:
    """
    Convert ReAct-style action to ToolExecutor dict format.
    """
    mapping = {
        "filter_data":    lambda a: {"tool": "filter",    "args": {"column": a["column"], "op": "==", "value": a["value"]}},
        "group_by":       lambda a: {"tool": "groupby",   "args": {"column": a["column"]}},
        "aggregate_sum":  lambda a: {"tool": "aggregate", "args": {"column": a["column"], "agg": "sum"}},
        "aggregate_mean": lambda a: {"tool": "aggregate", "args": {"column": a["column"], "agg": "mean"}},
        "sort_by":        lambda a: {"tool": "sort",      "args": {"column": a["column"], "ascending": a.get("order", "desc") == "asc"}},
        "top_k":          lambda a: {"tool": "topk",      "args": {"k": a["k"]}},
    }
    if action_name not in mapping:
        raise ValueError(f"Unknown action: {action_name}")
    return mapping[action_name](args)


# ================================================================
# 4. Context Manager (truncation / OOM prevention)
# ================================================================

class ContextManager:
    MAX_OBS_CHARS = 600

    def __init__(self, system_prompt: str, user_query: str):
        self._system = system_prompt
        self._query  = user_query
        self._turns  = []

    def build_prompt(self) -> str:
        history = "\n".join(self._turns)
        return f"{self._system}\n\nUser: {self._query}\n\n{history}".strip()

    def append(self, text: str):
        self._turns.append(text.strip())

    def append_observation(self, obs: str):
        if len(obs) > self.MAX_OBS_CHARS:
            obs = obs[:self.MAX_OBS_CHARS] + " ... [truncated]"
        self._turns.append(f"Observation: {obs}")


# ================================================================
# 5. ReAct Agent
# ================================================================

SYSTEM_PROMPT = """\
You are a data analysis agent. Use the following tools step by step to answer the user's query.

Dataset columns: date, year, month, city, region, product, category, revenue, units_sold, cost, profit

For each step output EXACTLY:
Thought: <your reasoning>
Action: <one of: filter_data, group_by, aggregate_sum, aggregate_mean, sort_by, top_k>
Action Input: <JSON dict of arguments>

When done, output:
Thought: I now have the final answer.
Final Answer: <your answer>
"""


class ReActAgent:
    def __init__(self, model, tokenizer, df: pd.DataFrame, max_turns: int = 5, max_new_tokens: int = 200):
        self.model         = model
        self.tokenizer     = tokenizer
        self.df            = df
        self.max_turns     = max_turns
        self.max_new_tokens = max_new_tokens

        self._stop_list = StoppingCriteriaList([
            ObservationStoppingCriteria(tokenizer),
            FinalAnswerStoppingCriteria(tokenizer),
        ])

    def _generate(self, prompt: str) -> str:
        inputs   = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        in_len   = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self._stop_list,
            )
        return self.tokenizer.decode(out[0][in_len:], skip_special_tokens=True)

    def run(self, query: str, verbose: bool = True) -> str:
        ctx = ContextManager(SYSTEM_PROMPT, query)
        executor   = ToolExecutor(self.df)
        pending    = []           # accumulate tool dicts before execute
        consec_err = 0

        for turn in range(1, self.max_turns + 1):
            if verbose:
                print(f"\n--- Turn {turn} ---")

            generated = self._generate(ctx.build_prompt())
            if verbose:
                print(generated)
            ctx.append(generated)

            # Check for final answer
            final = extract_final_answer(generated)
            if final:
                return final

            # Parse action
            parsed = parse_action_and_input(generated)
            if parsed is None:
                consec_err += 1
                ctx.append_observation("ERROR: Could not parse Action/Action Input. Please retry with correct format.")
                if consec_err >= 2:
                    return "Agent failed: repeated parse errors."
                continue

            action_name, args = parsed
            consec_err = 0

            # Execute via ToolExecutor
            try:
                tool_dict = dispatch(action_name, args)
                pending.append(tool_dict)

                # Only execute immediately for non-terminal tools that return a scalar,
                # OR for the final step in a group→aggregate→sort→topk chain.
                # Simpler: execute one step at a time using a fresh executor each turn.
                step_executor = ToolExecutor(self.df)
                step_result   = step_executor.execute(pending)
                obs = str(step_result.to_string(index=False)) if hasattr(step_result, "to_string") else str(step_result)
            except Exception as e:
                consec_err += 1
                obs = f"TOOL ERROR: {type(e).__name__}: {e}"
                if verbose:
                    print(f"⚠ {obs}")
                pending.pop()   # remove failed step
                if consec_err >= 2:
                    return f"Agent failed after tool error: {obs}"

            ctx.append_observation(obs)

        return "Agent reached max turns without a final answer."


# ================================================================
# 6. Model loader
# ================================================================

def load_react_model(base_model_id: str = "mistralai/Mistral-7B-v0.1", adapter_path: str | None = None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import sys
    base_model  = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-v0.1"
    adapter_dir = sys.argv[2] if len(sys.argv) > 2 else None
    csv_path    = sys.argv[3] if len(sys.argv) > 3 else "sales_data.csv"

    df = pd.read_csv(csv_path)
    model, tokenizer = load_react_model(base_model, adapter_dir)
    agent = ReActAgent(model, tokenizer, df, max_turns=5)

    for q in [
        "What is total revenue for 2022?",
        "List top 3 cities based on revenue for 2021",
    ]:
        print(f"\n{'='*60}\nQUERY: {q}\n{'='*60}")
        print("FINAL ANSWER:", agent.run(q))
