# CS F425 Deep Learning — Course Project
## Building an Autonomous AI Agent

> **Deadline:** April 18th, 12:00 PM  
> **Evaluation:** Live inference demo — no training during the slot

---

## File Structure

```
cs-f425-agent/
│
├── tool_executor.py              # Professor's tool engine (do not modify)
├── example_use_tool_executor.py  # Professor's usage example
│
├── run_pipeline.py               # Runs all 2000 trajectories through ToolExecutor
├── agent.py                      # Phase 1 — LLM agent (prompt + inference)
├── sft_toolalpaca.py             # Phase 2A — QLoRA fine-tuning
├── react_agent.py                # Phase 2B — ReAct execution loop
│
├── agent_trajectories_2k.json    # Training data (2000 query → action pairs)
├── sales_data.csv                # Sales dataset (10,000 rows)
│
└── requirements.txt              # Python dependencies
```

---

## Setup

### 1. Clone and open in Colab

```python
!git clone https://github.com/YOUR_USERNAME/cs-f425-agent.git
%cd cs-f425-agent
```

### 2. Install dependencies

```python
!pip install -q -r requirements.txt
```

### 3. Mount Google Drive (to save model weights)

```python
from google.colab import drive
drive.mount('/content/drive')

ADAPTER_PATH = '/content/drive/MyDrive/ML_project/sft_adapter'
```

### 4. Verify GPU

```python
import torch
print(torch.cuda.is_available())   # must be True
print(torch.cuda.get_device_name(0))
```

If `False` → `Runtime → Change runtime type → T4 GPU`

---

## Phase 1 — Data Analysis Agent

The agent takes a natural language query, generates a tool action sequence using an LLM, and executes it against the sales CSV via `ToolExecutor`.

### How the tools work

`ToolExecutor` accepts a list of dicts and executes them in sequence:

```python
import pandas as pd
from tool_executor import ToolExecutor

df = pd.read_csv('sales_data.csv')
executor = ToolExecutor(df)

result = executor.execute([
    {"tool": "filter",    "args": {"column": "year", "op": "==", "value": 2022}},
    {"tool": "groupby",   "args": {"column": "city"}},
    {"tool": "aggregate", "args": {"column": "revenue", "agg": "sum"}},
    {"tool": "sort",      "args": {"column": "revenue", "ascending": False}},
    {"tool": "topk",      "args": {"k": 3}},
])
print(result)
```

### The 3 action patterns in the training data

| Pattern | Steps | Count |
|---------|-------|-------|
| A | `filter_data` → `aggregate_sum` | 1319 |
| B | `filter_data` → `group_by` → `aggregate_sum` → `sort_by` → `top_k` | 651 |
| C | `group_by` → `aggregate_mean` | 30 |

### Run the pipeline (no model needed)

Executes all 2000 pre-written trajectories and writes results to a file:

```python
from run_pipeline import run_automated_pipeline
run_automated_pipeline('sales_data.csv', 'agent_trajectories_2k.json', 'results.txt')
```

> **Note:** The original `run_pipeline.py` was missing an `aggregate_mean` handler — this repo has the fix applied.

### Run the LLM agent

```python
import pandas as pd
from agent import load_model, run_agent

df = pd.read_csv('sales_data.csv')
load_model()  # loads Mistral-7B-Instruct, ~5 min on T4

result = run_agent("List top 3 cities by revenue for 2021", df)
print(result['actions'])
print(result['answer'])
```

After fine-tuning, pass the adapter:

```python
load_model(adapter_path=ADAPTER_PATH)
```

---

## Phase 2A — SFT Fine-Tuning

Fine-tunes `Mistral-7B-v0.1` on the 2000 trajectory examples using QLoRA. The model learns to map a plain English query to the correct `Thought / Action / Action Input` sequence.

### What it learns

```
Input:   "List top 3 cities by revenue for 2021"

Output:  Thought: I need to filter the data for the specified condition.
         Action: filter_data
         Action Input: {"column": "year", "value": 2021}

         Thought: I need to group the data by the specified column.
         Action: group_by
         Action Input: {"column": "city"}
         ...
         Final Answer: Based on the executed steps.
```

### Run training (~60–90 min on T4)

```python
import sft_toolalpaca as cfg
cfg.OUTPUT_DIR = ADAPTER_PATH

from sft_toolalpaca import train
model, tokenizer = train('agent_trajectories_2k.json')
```

Training logs will show loss dropping from ~1.5 → ~0.3–0.6 over 3 epochs.

### Tier-1 offline evaluation

Checks that the model generates the correct sequence of action names:

```python
from sft_toolalpaca import offline_eval
acc = offline_eval(model, tokenizer, 'agent_trajectories_2k.json', n=100)
# Target: > 70% format accuracy
```

---

## Phase 2B — ReAct Execution Loop

A pure Python multi-turn agent loop. No LangChain, no LlamaIndex.

### How one turn works

```
prompt + history  →  model.generate()
                  →  stops at "Observation:" token   ← custom StoppingCriteria
                  →  parse Action + Action Input
                  →  ToolExecutor.execute()
                  →  append real result as Observation
                  →  next turn
                  →  stops at "Final Answer:"
```

### The 5 required components

| Component | What it does |
|-----------|-------------|
| `ObservationStoppingCriteria` | Halts generation before model can hallucinate a tool result |
| `FinalAnswerStoppingCriteria` | Detects when the model is done |
| `parse_action_and_input()` | Robust JSON parser — handles malformed output from small models |
| `ContextManager` | Truncates observations to 600 chars to prevent OOM |
| Error recovery | Counts consecutive failures, terminates gracefully after 2 |

### Run the ReAct agent

```python
import pandas as pd
from react_agent import load_react_model, ReActAgent

df = pd.read_csv('sales_data.csv')

model, tokenizer = load_react_model(
    base_model_id='mistralai/Mistral-7B-v0.1',
    adapter_path=ADAPTER_PATH,
)

agent = ReActAgent(model, tokenizer, df, max_turns=5)
answer = agent.run("List top 3 cities by revenue for 2021", verbose=True)
print("Final Answer:", answer)
```

---

## Evaluation (Live Demo Slot)

Load saved weights — no training during the slot:

```python
# Phase 1
import pandas as pd
from agent import load_model, run_agent

df = pd.read_csv('sales_data.csv')
load_model(adapter_path=ADAPTER_PATH)

result = run_agent("YOUR HELD-OUT QUERY HERE", df)
print(result)
```

```python
# Phase 2
from react_agent import load_react_model, ReActAgent

model, tokenizer = load_react_model(adapter_path=ADAPTER_PATH)
agent = ReActAgent(model, tokenizer, df, max_turns=5)
print(agent.run("YOUR HELD-OUT QUERY HERE"))
```

---

## Dataset

`sales_data.csv` — 10,000 rows of sales records

| Column | Type | Example |
|--------|------|---------|
| date | string | 2023-12-06 |
| year | int | 2023 |
| month | int | 12 |
| city | string | Mumbai |
| region | string | West |
| product | string | D |
| category | string | Electronics |
| revenue | float | 15302.75 |
| units_sold | int | 60 |
| cost | float | 8177.77 |
| profit | float | 7124.98 |

Cities: Mumbai, Delhi, Bangalore, Chennai, Kolkata  
Regions: North, South, East, West  
Years: 2021, 2022, 2023

---

## Hardware

All training runs on free-tier GPU:
- Google Colab T4 (15GB VRAM)
- Kaggle T4/P100 (30 hrs/week, more reliable)

Model is loaded in 4-bit NF4 quantisation via `bitsandbytes`. LoRA adapter is ~50MB and saved to Google Drive.
