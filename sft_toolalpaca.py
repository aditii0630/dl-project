# ============================================================
# CS F425 Deep Learning — Phase 2A
# sft_toolalpaca.py  |  Fine-tune on agent_trajectories_2k.json
# ============================================================
# Training data is the professor's own trajectories file —
# 2000 examples, 3 action patterns, real sales columns.
#
# Model learns to output:
#   Thought: <reasoning>
#   Action: <tool_name>
#   Action Input: {"key": value}
#   ... (one Thought/Action/Action Input block per step)
# ============================================================

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
MODEL_ID    = os.getenv("MODEL_ID", "mistralai/Mistral-7B-v0.1")
OUTPUT_DIR  = "./sft_adapter"
HF_REPO     = None        # set to "username/repo" to push weights
MAX_SEQ_LEN = 512
BATCH_SIZE  = 2
GRAD_ACCUM  = 8           # effective batch = 16
LR          = 2e-4
EPOCHS      = 3
LORA_R      = 16
LORA_ALPHA  = 32
LORA_TARGET = ["q_proj", "v_proj", "k_proj", "o_proj"]

# ----------------------------------------------------------------
# Action string → Thought/Action/Action Input block
# ----------------------------------------------------------------

# Maps agent action strings to the ReAct-style multi-turn format.
# Each action string becomes one Thought + Action + Action Input block.

_THOUGHTS = {
    "filter_data":    "I need to filter the data for the specified condition.",
    "group_by":       "I need to group the data by the specified column.",
    "aggregate_sum":  "I need to compute the sum of the specified column.",
    "aggregate_mean": "I need to compute the mean of the specified column.",
    "sort_by":        "I need to sort the results by the specified column.",
    "top_k":          "I need to select the top entries from the results.",
}


def action_str_to_react_block(action_str: str) -> str:
    """
    "filter_data(column='year', value=2022)"
    →
    "Thought: I need to filter the data for the specified condition.
     Action: filter_data
     Action Input: {"column": "year", "value": 2022}"
    """
    import re
    name = action_str.split("(")[0].strip()
    thought = _THOUGHTS.get(name, "I need to execute the next step.")

    # Parse kwargs
    args_str = action_str[action_str.index("(") + 1: action_str.rindex(")")]
    kwargs = {}
    for m in re.finditer(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(-?\d+(?:\.\d+)?))", args_str):
        key = m.group(1)
        val = m.group(2) or m.group(3) or m.group(4)
        try:    val = int(val)
        except: pass
        kwargs[key] = val

    action_input = json.dumps(kwargs)
    return f"Thought: {thought}\nAction: {name}\nAction Input: {action_input}"


# ----------------------------------------------------------------
# Build training examples from trajectories
# ----------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a data analysis agent. Given a user query about sales data, \
output a sequence of Thought/Action/Action Input steps to answer it.

Dataset columns: date, year, month, city, region, product, category, revenue, units_sold, cost, profit

Available actions: filter_data, group_by, aggregate_sum, aggregate_mean, sort_by, top_k

After all steps, output:
Final Answer: <result description>"""


def build_example(entry: dict) -> dict:
    """Convert one trajectory entry into a training prompt + completion."""
    query   = entry["query"]
    actions = entry["actions"]

    # Build the multi-step ReAct completion
    blocks  = [action_str_to_react_block(a) for a in actions]
    completion = "\n\n".join(blocks) + "\n\nFinal Answer: Based on the executed steps."

    # Full text = system prompt + user query + assistant completion
    text = (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[USER]\n{query}\n\n"
        f"[ASSISTANT]\n{completion}"
    )
    return {"text": text, "completion": completion}


def load_trajectories(path: str) -> Dataset:
    with open(path) as f:
        data = json.load(f)
    examples = [build_example(e) for e in data]
    print(f"[SFT] Loaded {len(examples)} training examples")
    print(f"[SFT] Sample:\n{examples[0]['text'][:500]}\n...")
    return Dataset.from_list(examples)


# ----------------------------------------------------------------
# QLoRA model setup
# ----------------------------------------------------------------

def _is_bitsandbytes_4bit_healthy() -> bool:
    try:
        import bitsandbytes as bnb
        # Trigger a tiny 4-bit op to verify native CUDA path is actually usable.
        probe = torch.randn(128, device="cuda", dtype=torch.float16)
        bnb.functional.quantize_4bit(probe, quant_type="nf4")
        return True
    except Exception:
        return False


def load_base_model(model_id: str | None = None, use_4bit_preference: bool = True):
    model_id = model_id or MODEL_ID
    use_4bit = bool(use_4bit_preference and torch.cuda.is_available() and _is_bitsandbytes_4bit_healthy())

    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_gb < 28:
                raise RuntimeError(
                    "7B training requires either healthy 4-bit bitsandbytes or a larger GPU. "
                    f"Detected ~{total_gb:.1f} GB VRAM without usable 4-bit path. "
                    "Use A100/H100 runtime or fix bitsandbytes CUDA dependencies."
                )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET,
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    ))
    model.print_trainable_parameters()
    return model


# ----------------------------------------------------------------
# Train
# ----------------------------------------------------------------

def train(trajectories_path: str = "agent_trajectories_2k.json"):
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds    = load_trajectories(trajectories_path)
    use_4bit = bool(torch.cuda.is_available() and _is_bitsandbytes_4bit_healthy())
    model = load_base_model(use_4bit_preference=use_4bit)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # Only compute loss on [ASSISTANT] completions
    

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=training_args,
        
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    print("[SFT] Training...")
    trainer.train()

    print(f"[SFT] Saving adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if HF_REPO:
        trainer.model.push_to_hub(HF_REPO)
        tokenizer.push_to_hub(HF_REPO)

    print("[SFT] Done!")
    return trainer.model, tokenizer


# ----------------------------------------------------------------
# Offline (Tier-1) evaluation — format correctness
# ----------------------------------------------------------------

def offline_eval(model, tokenizer, trajectories_path: str, n: int = 100) -> float:
    """
    Check that the model outputs the correct Thought/Action/Action Input
    structure and that Action names match the gold sequence.
    """
    import re
    with open(trajectories_path) as f:
        data = json.load(f)[:n]

    correct = 0
    model.eval()
    for entry in data:
        query      = entry["query"]
        gold_names = [a.split("(")[0] for a in entry["actions"]]

        prompt = (
            f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
            f"[USER]\n{query}\n\n"
            f"[ASSISTANT]\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=300,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        pred_names = re.findall(r"Action:\s*(\w+)", gen)
        match = pred_names == gold_names
        if match: correct += 1

    acc = correct / n
    print(f"[Eval] Tier-1 accuracy: {acc:.1%} ({correct}/{n})")
    return acc


if __name__ == "__main__":
    model, tokenizer = train()
    offline_eval(model, tokenizer, "agent_trajectories_2k.json", n=100)
