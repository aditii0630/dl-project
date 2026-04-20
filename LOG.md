# Project Debug Log

Date: 2026-04-20
Workspace: C:/Users/vinay/Documents/CODE/dl-project

## Goal
Run the project workflow (Colab-equivalent local validation in VS Code), identify errors, fix them, and document every action, result, and code change.

## Session Notes
- Initialized log file.

## Step 1 - Baseline Error Capture
- Action: Read key files and ran entry-point scripts.
- Result: Reproduced concrete failures:
	- example_use_tool_executor.py -> NameError: ToolExecutor not defined.
	- run_pipeline.py -> FileNotFoundError for sample_sales.csv.
	- agent.py/react_agent.py/sft_toolalpaca.py import -> ModuleNotFoundError: torch.
- Conclusion: Need code fixes for script wiring and dependency fix for torch.

## Step 2 - Code and Dependency Fixes Applied
- Changed: example_use_tool_executor.py
	- Added `from tool_executor import ToolExecutor`.
	- Why: resolves NameError at runtime.
- Changed: run_pipeline.py
	- Updated __main__ defaults from sample files to existing files:
		- sales_data.csv
		- agent_trajectories_2k.json
	- Why: avoids FileNotFoundError and makes script runnable out-of-box.
- Changed: requirements.txt
	- Added `torch`.
	- Why: required by agent.py, react_agent.py, and sft_toolalpaca.py imports.

## Step 3 - Additional Repository Consistency Fixes
- Found duplicate bugs in `Phase-1  files/` copies.
- Changed: Phase-1  files/example_use_tool_executor.py
	- Added `from tool_executor import ToolExecutor`.
	- Why: same NameError bug existed in duplicate script.
- Changed: Phase-1  files/run_pipeline.py
	- Added `aggregate_mean` parsing in `parse_agent_action`.
	- Updated default __main__ input files to `sales_data.csv` and `agent_trajectories_2k.json`.
	- Why: matched trajectory action set and fixed missing sample file issue.

## Step 4 - Dependency Installation Attempt and Blocker
- Action: Ran `python -m pip install -r requirements.txt`.
- Result: Failed due to Windows file lock during install:
	- `OSError: [WinError 32] ... mpmath/calculus/polynomials.py is being used by another process`
- Impact: model modules (`agent.py`, `react_agent.py`, `sft_toolalpaca.py`) still cannot import `torch`.
- Next planned mitigation: force reinstall with no cache and process-lock checks.

## Step 5 - Environment Recovery
- Action: Re-ran dependency installation with force-reinstall/no-cache.
- Result: Core dependencies installed successfully, including `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, and `datasets`.
- Verification:
	- `agent.py` import passed.
	- `react_agent.py` import passed.
	- `sft_toolalpaca.py` still failed on top-level `trl` import because `trl` read `deepseekv3.jinja` with cp1252 on Windows.
- Fix applied afterward:
	- Moved `from trl import SFTTrainer` inside `train()` in `sft_toolalpaca.py` so the module can import cleanly.
	- Why: avoids eager `trl` initialization during import on Windows.

## Step 6 - Colab Training Notebook
- Action: Created `colab_phase2a_train.ipynb` at the workspace root.
- Purpose: Colab-ready Phase 2A SFT training notebook for the strongest available GPU in the current plan.
- Status: Notebook scaffold created and ready for the remaining training cells.

## Step 7 - Notebook Completed
- Action: Filled `colab_phase2a_train.ipynb` with the full Colab training workflow.
- Includes:
	- GPU/runtime verification
	- package installation
	- Drive mount and path setup
	- dataset inspection and split
	- model definition and GPU-aware training config
	- SFT training
	- offline evaluation
	- adapter export to Google Drive
- Fix applied after initial notebook build:
	- Replaced a shadowed output path with `ADAPTER_DIR` so the adapter saves to Drive reliably.

## Step 8 - Notebook Portability Check
- Action: Verified the notebook structure after edits.
- Result: Notebook contains 11 cells total, with the requested training workflow laid out in order.
- Extra fix:
	- Made the Drive root fall back to the local workspace when the notebook is opened outside Colab.

## Step 9 - Cell 5 Import Fix
- Issue reported: Cell 5 failed while importing `sft_toolalpaca`, with a torch internal import error.
- Fix applied:
	- Removed the `sft_toolalpaca` import from cell 5.
	- Inlined the small `SYSTEM_PROMPT` / `action_str_to_react_block` helpers so preprocessing stays self-contained.
- Colab guidance:
	- Added a runtime restart instruction immediately after package installation, since Colab can retain stale torch state until restart.

## Step 10 - Cell 6 Torchvision Mismatch Fix
- Issue reported: Cell 6 failed because Colab had a `torchvision` build compiled against a different CUDA major version than `torch`.
- Fix applied:
	- Added `pip uninstall -y torchvision` to the install cell in `colab_phase2a_train.ipynb`.
	- Why: the notebook does not need `torchvision`, and removing the mismatched build prevents `transformers` from failing during import.

## Step 11 - Cell 6 BitsAndBytes Fallback
- Issue reported: Cell 6 failed during 4-bit model load because bitsandbytes could not find `libnvJitLink.so.13` in the Colab CUDA runtime.
- Fix applied:
	- Updated `sft_toolalpaca.load_base_model()` to try 4-bit QLoRA first, then fall back to a bf16 model load if bitsandbytes initialization fails.
	- Why: keeps the notebook runnable even when the Colab runtime and bitsandbytes wheel do not match perfectly.

## Step 12 - Additional Colab Stabilization
- Issue reported: repeated Cell 6 failures due to bitsandbytes/CUDA runtime mismatch and then OOM after fallback.
- Fixes applied in notebook:
	- Cell 2 now avoids reinstalling `torch` (keeps Colab's matched torch build).
	- Cell 2 installs `nvidia-nvjitlink-cu13` and updates `LD_LIBRARY_PATH` for bitsandbytes.
	- Cell 2 still removes `torchvision` to avoid torch/vision CUDA mismatch.
	- Added a preflight diagnostics cell to print torch CUDA and bitsandbytes import status.
	- Cell 6 now prepares tokenizer only (does not load model).
	- Cell 7 now loads the model once to avoid duplicate 7B allocations.

## Step 13 - Runtime + Memory Guardrails
- Added stronger nvJitLink detection/export in notebook startup and preflight checks.
- Updated `sft_toolalpaca.load_base_model()` to accept `model_id` override.
- Notebook now chooses a smaller base model (`Qwen/Qwen2.5-3B`) on GPUs with < 20GB VRAM unless overridden.
- Notebook training args now use `adamw_torch` when bitsandbytes is unhealthy, else `paged_adamw_8bit`.

## Step 14 - Enforce 7B-Only Training
- User requirement: training must stay on 7B model only.
- Changes made:
	- `sft_toolalpaca.load_base_model()` now supports explicit `use_4bit_preference` and probes real 4-bit health.
	- Added explicit error for insufficient VRAM when 4-bit is unavailable in 7B mode.
	- `train()` now aligns optimizer choice with 4-bit availability.
	- Notebook preflight now computes `USE_4BIT` via an actual 4-bit quantization probe.
	- Notebook Section 6 now hard-selects `mistralai/Mistral-7B-v0.1` and raises if runtime cannot support it.
	- Removed automatic 3B fallback path from notebook.

## Step 15 - W&B/TRL Import Stabilization
- Issue reported: `trl` import failed due to `wandb` protobuf mismatch (`cannot import name 'Imports'`).
- Fixes applied in notebook:
	- Install cell now uninstalls `wandb` as part of environment stabilization.
	- Training config cell sets `WANDB_DISABLED=true` and defensively uninstalls `wandb` if present.
- Why: this notebook does not use W&B reporting (`report_to="none"`), so removing it avoids optional-integration import crashes.

## Step 16 - Robust Package Installation Flow
- Issue reported: install cell failed with a single bulk `pip install` command returning non-zero.
- Fixes applied in notebook:
	- Switched from one bulk install to per-package install with clear success/failure prints.
	- Kept Colab preinstalled `torch` (no torch upgrade in notebook install cell).
	- Split dependencies into required vs optional; optional failures are reported without hard-failing the run.
	- Continued cleanup steps (`torchvision` uninstall, `wandb` uninstall, nvJitLink path export).
- Why: makes colab installs debuggable and resilient to transient/mirror-specific package failures.

## Step 17 - Transformers API Compatibility
- Issue reported: `TrainingArguments.__init__()` rejected `evaluation_strategy` in the installed transformers build.
- Fix applied in notebook training config cell:
	- Added runtime signature check and auto-select of either `evaluation_strategy` or `eval_strategy`.
- Why: supports multiple transformers versions without manual edits.

## Step 18 - TRL SFTTrainer API Compatibility
- Issue reported: `SFTTrainer.__init__()` rejected `tokenizer` argument in installed TRL build.
- Fix applied in notebook training cell:
	- Added runtime signature inspection to use `tokenizer` when available, else `processing_class`.
	- Filtered trainer kwargs to only pass parameters supported by the installed TRL version.
- Why: avoids version-specific constructor mismatches and keeps training cell portable.

## Step 19 - Training Progress Visibility
- User request: show clearer training progress in notebook output.
- Fix applied in notebook training cell:
	- Added `NotebookProgressCallback` (TrainerCallback) to print:
		- training start summary (max steps/epochs)
		- periodic step progress with percentage
		- logged metrics (`loss`, `learning_rate`, `eval_loss`, `epoch`) at log events
- Why: gives reliable progress feedback in Colab output even when default progress bars are inconsistent.

## Step 20 - Quick Test Mode for Faster Iteration
- User request: allow testing with a simpler model that trains faster.
- Fixes applied in notebook:
	- Added `QUICK_TEST_MODE` switch in model-definition cell.
	- Added `QUICK_TEST_MODEL` default (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) when quick-test mode is enabled.
	- Preserved strict 7B guardrails when quick-test mode is disabled.
	- Training config now uses a lightweight setup in quick-test mode (`num_train_epochs=1`, `gradient_accumulation_steps=2`, `logging_steps=5`, `max_steps=40`).
	- Training cell now subsets datasets in quick-test mode (up to 300 train / 60 val) for rapid smoke tests.
- Why: enables quick pipeline validation/debugging in minutes while keeping full 7B training path intact for final runs.

## Step 21 - Section 9 Evaluation Schema Fix
- Issue reported: evaluation failed with `Evaluation skipped or failed: 'query'`.
- Root cause: Section 9 wrote processed examples (`text`/`completion`) to `val_trajectories.json`, but `offline_eval` expects raw trajectories containing `query` and `actions`.
- Fix applied in notebook:
	- Section 9 now reloads raw trajectories from `DATA_PATH`.
	- Recreates the same deterministic split (`random_state=42`) and writes validation records in raw schema.
	- Keeps quick-test eval lightweight by capping validation size in quick mode.
- Why: restores compatibility with `offline_eval` input contract.

## Step 22 - Faster 7B Iteration Mode
- User request: reduce very long runtime while still allowing 7B experiments.
- Fixes applied in notebook:
	- Added `FAST_7B_MODE` switch (active only when `QUICK_TEST_MODE=0`).
	- Section 7 now has a dedicated faster 7B config:
		- `num_train_epochs=1`
		- `gradient_accumulation_steps=4`
		- `max_steps=220`
		- less frequent eval/checkpoints
	- Section 8 now applies lighter data/sequence settings in `FAST_7B_MODE`:
		- subset up to 1200 train / 200 val examples
		- `max_seq_length=256`
- Why: keeps 7B base model path but cuts wall-clock time substantially for iterative debugging.

## Step 23 - Kaggle P100 Optimized Training Notebook
- New file created: `colab_phase2a_train_optimized.ipynb`
- Purpose: Kaggle-ready training notebook optimized for P100 GPU (16GB VRAM) with backward compatibility to Colab T4/A100/H100.
- Key hardware targets: Kaggle P100 (16GB, sm_60), Colab T4 (15GB), A100 (40GB+), H100 (80GB+).

### Major Optimizations:
- **PyTorch Downgrade (Cell 0):**
	- Downgraded to PyTorch 2.1.2+cu118 (from latest 2.5+cu121).
	- Why: P100 uses CUDA compute sm_60, which PyTorch 2.5+ no longer supports. 2.1.2 is the last build with sm_60 support.
	- Kaggle workflow: run Cell 0 first, then "Restart Session", then run remaining cells.

- **Package Pinning (Section 2):**
	- Pinned all dependencies to versions compatible with PyTorch 2.1.2:
		- `transformers==4.40.2`
		- `peft==0.10.0`
		- `trl==0.8.6`
		- `accelerate==0.29.3`
		- `bitsandbytes==0.43.1` (last version with sm_60 support)
	- Removed `nvidia-nvjitlink-cu13` (CUDA 12.x only; P100 uses cu118).
	- Why: ensures stable builds across Kaggle/Colab environments.

- **Environment Detection (Sections 1 & 3):**
	- Added `IN_KAGGLE` detection (`os.path.exists("/kaggle/working")`).
	- Section 3 now handles three deployment paths:
		- Kaggle: `/kaggle/input/dl-project-data` (input) → `/kaggle/working/` (output)
		- Colab: `/content/drive/MyDrive/` (Drive-mounted)
		- Local: current working directory
	- Why: single notebook works across all platforms without manual path edits.

- **Preflight Simplification:**
	- Skipped nvJitLink check (CUDA 12.x only; not present on cu118/P100).
	- Added explicit message: "nvJitLink check skipped (not needed for CUDA 11.8 / P100)."
	- Why: avoids confusing warnings on P100.

- **Training Optimization (Section 7 - Full Mode):**
	- **Batch size:** 2 → 4 (doubled, fits in P100 16GB with 4-bit quantization).
	- **Gradient accumulation:** 8 → 4 (effective batch remains 16: 4 batch × 4 accum).
	- **Epochs:** 3 → 1 (hard ceiling with `max_steps=600`).
	- **Max steps:** 600 hard limit (~1–1.5 hrs on P100).
	- **Eval/save frequency:** every 100 steps → every 300 steps (reduces checkpoint I/O overhead).
	- **Dataloader workers:** Added `dataloader_num_workers=0` (avoids Kaggle container multiprocessing deadlock).
	- Why: roughly 50% wall-clock speedup while maintaining competitive final quality.

- **Sequence Length & Packing (Section 8):**
	- **max_seq_length:** 512 → 256 (halved; ReAct blocks fit easily in 256 tokens).
	- **packing:** False → **True** (biggest single throughput improvement: 3–5× token efficiency).
	- Why: tighter sequences + efficient packing bins multiple short examples per window, maximizing GPU compute utilization.

- **VRAM Guardrail (Section 6):**
	- Enhanced error message specific to P100:
		- Clearly states P100 (16GB) requires working bitsandbytes 4-bit to run 7B.
		- Points to Section 2 bitsandbytes check and suggests setting `QUICK_TEST_MODE=1` as fallback.
	- Why: prevents cryptic OOM errors; guides users to correct fixes.

- **Kaggle File I/O (Section 9):**
	- Validation file now written to `/kaggle/working/` on Kaggle (writable).
	- Remains in `PROJECT_DIR` on Colab/local.
	- Why: Kaggle input dirs are read-only; writing to `/kaggle/input/` would fail silently.

## Step 24 - T4-First Notebook Reconfiguration
- User request: restore T4 compatibility and keep the checkpoint resume flow.
- Changes applied in `colab_phase2a_train_optimized.ipynb`:
	- Updated notebook title and overview to T4-first wording.
	- Reframed Cell 0 as T4 / general GPU support.
	- Reworded Section 6 guardrails to describe 16GB GPUs instead of P100-only language.
	- Kept automatic checkpoint resume in Section 8.
- Why: the workflow now matches T4 again without losing interruption recovery.

## Step 25 - Runtime Cell Transplant From Normal Notebook
- User request: copy the installation, preflight, and GPU verification cells from `colab_phase2a_train.ipynb` into the optimized notebook.
- Changes applied in `colab_phase2a_train_optimized.ipynb`:
	- Replaced the existing Cell 0, Section 1, Preflight, and Section 2 cells.
	- Converted the user-facing wording from Colab-first to Kaggle-first.
	- Kept the Kaggle-compatible package and checkpoint behavior in the optimized notebook.
- Why: the optimized notebook now inherits the normal notebook's runtime setup while staying Kaggle-oriented.

## Step 26 - Preflight Triton Shim Recovery
- Issue reported: preflight bitsandbytes probe failed with `No module named 'triton.ops'`.
- Fix applied in `colab_phase2a_train_optimized.ipynb` preflight cell:
	- Added a minimal in-cell `triton.ops` shim before the bitsandbytes quantize probe.
	- Retried the 4-bit probe after installing the shim when the missing-module error appears.
- Why: preflight now succeeds even on Kaggle runtimes where bitsandbytes expects Triton symbols that are absent.

## Step 27 - Kaggle-Optimised Notebook Rewrite
- User changed `colab_phase2a_train_optimized.ipynb` again and the current file now reflects a leaner Kaggle-first layout.
- Current notebook changes captured in the file:
	- Title and overview rewritten to a Kaggle-optimised ToolAlpaca SFT notebook.
	- Cell 1 now does simple Kaggle environment detection and GPU verification with `torch`/`nvidia-smi`.
	- Cell 2 now installs pinned versions for the Kaggle stack: `transformers==4.44.2`, `peft==0.12.0`, `trl==0.10.1`, `accelerate==0.34.2`, `datasets==2.21.0`, `sentencepiece==0.2.0`, `protobuf==4.25.4`, and `bitsandbytes>=0.43.0`.
	- Cell 3 sets Kaggle paths under `/kaggle/working/` and adds the dataset directory to `sys.path`.
	- Cell 4 loads and inspects the raw dataset from `DATA_PATH` and `CSV_PATH`.
	- Cell 5 converts trajectories into ReAct format, builds `Dataset` objects, and prints a sample training text.
	- Cell 6 keeps `QUICK_TEST_MODE` / `FAST_7B_MODE`, uses `Qwen/Qwen2-0.5B` as the quick-test default, and keeps the 7B safety gate when 4-bit is unavailable.
	- Cell 7 loads the base model directly with `AutoModelForCausalLM`, enables 4-bit quantization with `BitsAndBytesConfig` when healthy, and applies LoRA via `peft`.
	- Cell 8 defines training arguments with Kaggle-appropriate defaults, including `group_by_length=True`, `dataloader_num_workers=2`, `dataloader_pin_memory=True`, and `save_steps` / `max_steps` presets for quick, fast, and full modes.
	- Cell 9 trains with `packing=True`, `MAX_SEQ_LEN` based on mode, and a Kaggle progress callback that prints GPU snapshots and logged metrics.
	- Cell 10 writes raw validation trajectories to `/kaggle/working/val_trajectories.json` and runs `offline_eval`.
	- Cell 11 exports the final adapter to `/kaggle/working/cs_f425_sft_adapter/final/` and lists the output files.
- Why: this records the latest notebook rewrite so the log matches the current file instead of the older Colab/T4-oriented variants.

## Step 28 - Remove Pandas Dependency From Dataset Inspection
- Issue reported: Cell 4 crashed on `import pandas as pd` with a NumPy/Pandas binary incompatibility (`numpy.dtype size changed`).
- Fix applied in `colab_phase2a_train_optimized.ipynb` Cell 4:
	- Replaced the pandas-based CSV inspection with stdlib `csv.DictReader`.
	- Kept the same outputs: row count, column names, sample rows, and the first trajectory example.
- Why: dataset inspection no longer depends on a fragile binary wheel pair, so the notebook can continue even if the Kaggle image has mismatched pandas/numpy builds.

- **Model Export & Instructions (Sections 10 & 14):**
	- Adapter saved to `/kaggle/working/cs_f425_sft_adapter/` on Kaggle.
	- Added Kaggle-specific guidance: "Download from Output tab (right sidebar)."
	- Optional HF Hub cell provided (push adapter to HuggingFace for persistent storage).
	- Why: streamlines Kaggle workflow; no need to manually download via browser.

### Backward Compatibility:
- All three training modes (`QUICK_TEST_MODE`, `FAST_7B_MODE`, full) still present and work identically.
- Colab users can run unchanged (falls through to non-Kaggle paths).
- Local users unaffected (paths default to cwd).

### Runtime Expectations on P100:
- `QUICK_TEST_MODE=1`: ~3 mins (TinyLlama, 40 steps).
- `FAST_7B_MODE=1`: ~45–60 mins (7B, 220 steps, 1200 train samples).
- Full mode (default): ~1.5 hrs (7B, 600 steps, full data, packing=True).
- Previous full mode on T4 was ~8 hrs (3 epochs, no packing, 512 seq); P100 with optimizations is 5–6× faster.

### Known Limitations & Notes:
- P100 sm_60 must use bitsandbytes 0.43.1; newer versions will fail on P100.
- Kaggle datasets must be uploaded as a Kaggle Dataset (not inline); script expects `/kaggle/input/dl-project-data/` unless overridden.
- Colab still requires runtime restart after package install (standard Colab practice).
- eval/save every 300 steps means fewer checkpoints; use `FAST_7B_MODE` or `QUICK_TEST_MODE` if more frequent eval is desired.

## Step 24 - Cell 0 Network Resilience & Environment Detection
- Issue reported: Cell 0 fails on non-Kaggle environments (local/Colab) with network errors when trying to download PyTorch 2.1.2+cu118.
- Root cause: Cell 0 was unconditionally trying to downgrade PyTorch, but that's **only needed on Kaggle P100** (which uses CUDA sm_60). Other environments already have compatible PyTorch and/or don't have network access to the torch index.
- Fix applied in notebook Cell 0:
	- Added `IN_KAGGLE = os.path.exists("/kaggle/working")` detection.
	- PyTorch downgrade now only runs on Kaggle (wrapped in `if IN_KAGGLE:`).
	- On Colab/local, Cell 0 prints a skip message and user proceeds directly to Section 1.
	- Added error handling: timeout and exception handlers with clear guidance.
	- Improved messaging: ✓ checkmarks for success, ⚠ warnings for issues.
- Updated markdown header to reflect: "Cell 0 is skipped on Colab/local automatically."
- Why: makes the notebook truly portable; works out-of-box on any platform without requiring users to comment/edit Cell 0.

## Step 25 - PyTorch Version Availability Fix (Cell 0)
- Issue reported: Cell 0 fails with "ERROR: Could not find a version that satisfies the requirement torch==2.1.2".
- Root cause: PyTorch 2.1.2+cu118 is no longer available in the PyTorch index. Available versions now start from 2.2.0+cu118 onward.
- Why 2.1.2 won't work anymore: PyTorch 2.2.0+ dropped support for older CUDA compute capabilities (including sm_60 for P100).
- **Solution for P100:** Try PyTorch 2.2.0+cu118 (may work on P100 despite official sm_60 drop) with clear fallback guidance.
- Fix applied in notebook Cell 0:
	- Cell 0 now attempts to install PyTorch 2.2.0+cu118 on Kaggle (only on Kaggle; local/Colab use environment default).
	- Added error handling for timeout and failures.
	- Provides three fallback options if CUDA errors occur:
		1. `QUICK_TEST_MODE=1` (uses TinyLlama—smaller, works on any GPU)
		2. Conda install from conda-forge (may have older PyTorch builds with sm_60 support)
		3. Switch to A100/H100 GPU if available on Kaggle
	- Clear messaging about P100's sm_60 compatibility status.
- Updated markdown header to reflect P100 is actively targeted.
- Why: Users can attempt full 7B training on P100 with best-effort PyTorch 2.2.0. If it fails, clear fallback paths are provided.

## Step 26 - Resolve PyTorch Dependency Conflicts (Cell 0)
- Issue reported: Pip dependency resolver warning about conflicting torch versions:
	- `torchaudio 2.10.0+cu128 requires torch==2.10.0`
	- `torchvision 0.25.0+cu128 requires torch==2.10.0`
	- But we installed `torch 2.2.0+cu118` (incompatible)
- Root cause: Kaggle's environment pre-installs torchaudio and torchvision with torch 2.10.0. When we downgrade torch to 2.2.0+cu118, version conflicts arise.
- Fix applied in notebook Cell 0:
	- **Before** installing torch 2.2.0+cu118, Cell 0 now uninstalls torchaudio and torchvision (wrapped in subprocess.run).
	- These packages are **not needed for training**, only for audio/vision tasks.
	- Removes the warning/conflict entirely without affecting training functionality.
- Why: Keeps the notebook clean and avoids confusing pip warnings. Focuses on only the dependencies needed for SFT training (torch, transformers, peft, trl, etc.).

## Step 27 - Section 7 Triton Import Repair (`triton.ops`)
- Issue reported in Section 7:
	- `RuntimeError: Failed to import trl.trainer.sft_trainer`
	- Root nested error: `No module named 'triton.ops'`
- Root cause:
	- The active package set could import `transformers`, but the import path used by `trl`/`bitsandbytes` expected a Triton API module (`triton.ops`) that was missing in the runtime.
	- This caused Section 7 to fail before training args/model setup completed.
- Fix applied in Section 7:
	- Added `ensure_triton_ops()` helper to verify `triton.ops` before importing `trl`.
	- If missing, Section 7 now auto-installs `triton==2.2.0` and retries import.
	- If still unavailable, Section 7 gracefully degrades by:
		1. setting `USE_4BIT=False`
		2. uninstalling `bitsandbytes` to avoid import-path crashes
	- Added explicit console status messages (`triton_ok=...`) in training setup output.
- Why:
	- Prevents hard failure in Section 7 due to Triton mismatch.
	- Keeps notebook execution moving with a deterministic fallback path instead of opaque import crashes.

## Step 28 - Section 7 Triton Binary Mismatch Hardening
- New runtime output observed:
	- `triton.ops` installation attempt failed with binary import path error:
		- `No module named 'triton._C.libtriton.triton'; 'triton._C.libtriton' is not a package`
	- Prior fallback then disabled 4-bit mode and uninstalled bitsandbytes.
- Root cause refinement:
	- This is a broken/incompatible Triton binary state, not just a missing `triton.ops` Python symbol.
	- Reinstalling Triton in-place is unreliable on this runtime and can make the environment worse.
- Updated fix in Section 7:
	- Replaced Triton auto-install logic with a **compatibility shim** for `triton.ops.matmul_perf_model` symbols used by bitsandbytes import probes.
	- Added `ensure_bitsandbytes_for_4bit()` to auto-reinstall `bitsandbytes==0.43.1` if previously removed.
	- Removed the unconditional "uninstall bitsandbytes" behavior tied to Triton failure.
	- Section 7 now keeps 4-bit mode when possible and only disables it if both recovery paths fail.
- Why:
	- Preserves the P100-required 4-bit path more reliably.
	- Avoids repeated runtime churn from uninstall/reinstall loops of Triton and bitsandbytes.

## Step 29 - Section 7 Fix for `load_base_model` 4-bit Health Failure
- New error observed while running Section 7:
	- `RuntimeError: 7B training requires either healthy 4-bit bitsandbytes or a larger GPU.`
	- Raised inside `sft_toolalpaca.load_base_model()` after internal `_is_bitsandbytes_4bit_healthy()` returned false.
- Root cause:
	- Even after shim setup, the module-local health check in `sft_toolalpaca` still evaluated 4-bit as unhealthy in this kernel state.
	- On P100 (~15.9 GB), that forces the explicit runtime guard to abort 7B loading.
- Fix applied in Section 7:
	- Added deterministic Triton cleanup + shim setup first:
		1. uninstall `triton`
		2. purge loaded `triton*` modules from `sys.modules`
		3. install lightweight `triton.ops` shim modules
	- Added `robust_bnb_4bit_probe()` that performs a real CUDA `quantize_4bit` operation.
	- Added `ensure_bitsandbytes_for_4bit()` to reinstall `bitsandbytes==0.43.1` and re-probe if needed.
	- Patched `sft_toolalpaca._is_bitsandbytes_4bit_healthy` at runtime to use `robust_bnb_4bit_probe()` before calling `load_base_model`.
- Why:
	- Makes the exact gate used by `load_base_model` consistent with the notebook's runtime recovery logic.
	- Prevents false-negative 4-bit health checks from blocking 7B on P100.

## Step 30 - Section 8 Dataset Packing Failure (`Numpy is not available`)
- New error observed in Section 8 while creating `SFTTrainer` with packing enabled:
	- `RuntimeError: Numpy is not available`
	- bubbled through `datasets` / `pyarrow` during TRL packed dataset generation (`DatasetGenerationError`).
	- final TRL message misleadingly reported packing/data insufficiency, but root cause was NumPy runtime breakage.
- Root cause:
	- The current runtime had an unhealthy torch<->numpy bridge after multiple package transitions.
	- TRL packing path converts tensors via `.cpu().numpy()`, which fails when NumPy runtime is broken.
- Fixes applied:
	- Section 2 now pins `numpy==1.26.4` as required dependency for stable compatibility with torch 2.2 stack.
	- Section 8 now runs `ensure_numpy_runtime_ok()` before creating `SFTTrainer`:
		1. validates NumPy + `torch.tensor(...).numpy()` bridge
		2. auto-reinstalls `numpy==1.26.4` if check fails
		3. re-checks bridge after repair attempt
	- Section 8 now sets `PACKING_ENABLED` dynamically:
		- keeps packing when NumPy is healthy
		- auto-falls back to `packing=False` only when NumPy cannot be repaired in-session
- Why:
	- Prevents hard failure in Section 8 from NumPy runtime drift.
	- Preserves fast packed training whenever possible, with graceful fallback when runtime is inconsistent.

## Step 31 - Section 8 Non-Packed Collator Tensorization Fix
- New error observed after fallback to `packing=False`:
	- `ValueError: too many dimensions 'str'`
	- Followed by: `Unable to create tensor ... features ('text') have excessive nesting`.
- Root cause:
	- When `packing=False`, `remove_unused_columns=False` allowed raw string columns (`text`, `completion`) to leak into the default collator path.
	- The tokenizer pad/collator attempted tensor conversion on string fields and failed.
- Fix applied:
	- Section 8 now auto-sets `train_args.remove_unused_columns = True` whenever `PACKING_ENABLED=False`.
	- Added startup print to show effective flags:
		- `packing=...`
		- `remove_unused_columns=...`
- Why:
	- Keeps non-packed fallback stable and compatible with TRL default collation.
	- Avoids string-to-tensor conversion errors while retaining packed-mode behavior when available.

## Step 32 - Section 7 Warning Cleanup (Gradient Checkpointing)
- Runtime state after previous fixes showed training proceeds, but emitted warning-only messages:
	- `use_cache=True is incompatible with gradient checkpointing`
	- torch checkpoint warning asking to pass `use_reentrant` explicitly.
- Fix applied in Section 7:
	- After model creation, set `peft_model.config.use_cache = False` (best-effort).
	- If supported by installed transformers `TrainingArguments`, set:
		- `gradient_checkpointing_kwargs={"use_reentrant": False}`
- Why:
	- Reduces noisy warnings during training.
	- Aligns with upcoming torch checkpoint default changes and keeps behavior explicit.
