"""
GRPO Training Script with Analysis-Based Fixes

Key fixes applied based on training vs eval flow analysis:

1. Turn Index: Uses session-derived turn_idx (like eval) instead of dataset counter
2. Candidate Selection: Prefers valid-action candidates when rewards tie
3. Input Sanitization: Cleans completions before env.simulate() and env.step()
4. Logging: Separates simulation_done vs commit_done, adds debugging fields
5. Enhanced Debugging: Logs both dataset and session turn indices for comparison

These changes should:
- Make turn_idx progress naturally (0 → 1 → 2 → ...) instead of sticking at 0
- Reduce parsing failures that cause immediate episode termination
- Align training logs with eval logs for easier comparison

For debugging, temporarily set:
- cfg.num_generations = 1
- cfg.per_device_train_batch_size = 1  
- cfg.gradient_accumulation_steps = 1
This makes training behave exactly like eval (one completion per step).
"""

# # put these at the very top of the file
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.9'

# import time, subprocess, GPUtil
import os, time, subprocess, GPUtil


def wait_for_healthy_gpus(required=2, max_load=0.30, max_mem=0.30, poll=10):
    """Return list of available GPU ids."""
    while True:
        candidates = GPUtil.getAvailable(
            order="memory", limit=16, maxLoad=max_load, maxMemory=max_mem
        )
        if len(candidates) >= required:
            return candidates[:required]
        print(f"[GPU picker] Not enough available GPUs; retrying in {poll}s...")
        time.sleep(poll)


def pick_gpus_and_budgets(required=2, safety_frac=0.80):
    """
    Picks 'required' healthy GPUs (absolute ids), sets CUDA_VISIBLE_DEVICES to them,
    and returns a max_memory dict keyed by *local* ordinals {0: 'XXGiB', 1: 'YYGiB', ...}.
    """
    abs_ids = wait_for_healthy_gpus(required=required)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, abs_ids))
    print(
        "[GPU picker] Using absolute GPUs:",
        abs_ids,
        "(local ordinals will be 0..",
        len(abs_ids) - 1,
        ")",
    )

    # Query total memory for each *absolute* id
    totals_mib = []
    for abs_id in abs_ids:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(abs_id),
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        ).strip()
        totals_mib.append(int(out))

    # Build local-ordinal budgets (visible devices reindexed to 0..N-1)
    max_memory = {
        i: f"{int((tot_mib/1024.0) * safety_frac)}GiB"
        for i, tot_mib in enumerate(totals_mib)
    }
    print("[GPU picker] max_memory per local device:", max_memory)
    return max_memory, len(abs_ids)


# Ask for 3 GPUs (falls back to however many it can get >=1)
MAX_MEMORY, NUM_VISIBLE = pick_gpus_and_budgets(required=2, safety_frac=0.80)

# (Optional) allocator hygiene helps fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:128,garbage_collection_threshold:0.9"
)


import torch

# Single process with model sharding (Option A)
LOCAL_RANK = 0
WORLD_SIZE = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pathlib, sys

# add src/ so we can import YAML loader utils if needed later
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

import json, torch, requests, re, copy
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from lab_http_env import LABHTTPEnv  # ← our tiny client
from pathlib import Path
from torch.utils.data import Dataset
from dataclasses import dataclass
from transformers.utils import ModelOutput
import torch.nn as nn
from contextlib import contextmanager


@dataclass
class CausalLMOutputWithValue(ModelOutput):
    logits: torch.FloatTensor = None
    value: torch.FloatTensor = None  # required by GRPO


class DictOutputWrapper(nn.Module):
    """
    Wrapper that keeps return_dict=True and presents a model-like surface
    (config, generate, warnings_issued, etc.) to TRL.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)

        # Ensure the underlying model has a dict we can mutate in place
        if (
            not hasattr(self.base_model, "warnings_issued")
            or self.base_model.warnings_issued is None
        ):
            self.base_model.warnings_issued = {}

    def _pm(self):
        # underlying HF model that actually has the embedding utils
        return getattr(self.base_model, "pretrained_model", self.base_model)

    @contextmanager
    def _temp_gen_mode(self):
        pm = self._pm()
        # remember current flags
        old_uc = getattr(pm.config, "use_cache", False)
        # some models expose .is_gradient_checkpointing, some store a flag
        had_gc_attr = hasattr(pm, "is_gradient_checkpointing")
        try:
            was_gc = bool(getattr(pm, "is_gradient_checkpointing", False))
        except Exception:
            was_gc = False

        # switch: turn OFF GC, turn ON cache
        try:
            if was_gc and hasattr(pm, "gradient_checkpointing_disable"):
                pm.gradient_checkpointing_disable()
        except Exception:
            pass
        pm.config.use_cache = True

        try:
            yield
        finally:
            # restore flags
            pm.config.use_cache = old_uc
            try:
                if was_gc and hasattr(pm, "gradient_checkpointing_enable"):
                    pm.gradient_checkpointing_enable()
            except Exception:
                pass

    @property
    def warnings_issued(self):
        # TRL mutates this in place (model.warnings_issued["..."]=True)
        return self.base_model.warnings_issued

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        out = self.base_model(*args, **kwargs)
        if isinstance(out, tuple):
            logits, value = out[:2]
            return CausalLMOutputWithValue(logits=logits, value=value)
        return out

    # Forward common generation utilities
    def generate(self, *args, **kwargs):
        # ensure HF gets the hint too
        kwargs.setdefault("use_cache", True)
        with torch.no_grad():
            with self._temp_gen_mode():
                return self._pm().generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        pm = self._pm()
        if hasattr(pm, "prepare_inputs_for_generation"):
            return pm.prepare_inputs_for_generation(*args, **kwargs)
        # optional: fall back to base if present
        if hasattr(self.base_model, "prepare_inputs_for_generation"):
            return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
        raise AttributeError("prepare_inputs_for_generation not available")

    def get_input_embeddings(self):
        return self._pm().get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self._pm().set_input_embeddings(new_embeddings)

    def resize_token_embeddings(self, *args, **kwargs):
        return self._pm().resize_token_embeddings(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def __getattr__(self, name):
        """
        First try nn.Module's own attribute resolution (modules, params, buffers, etc.).
        Only if that fails, delegate to the underlying base_model.
        """
        try:
            return super().__getattr__(name)  # nn.Module.__getattr__
        except AttributeError:
            return getattr(self.base_model, name)


# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

ENTROPY_PCTL = 0.5
BATCH_SIZE = 8
MAX_TOKENS = 512  # match eval first

# No need to download dataset - server loads local data
print("Using local dataset loaded by server...")

# 1. Model
tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Log default values before applying fixes
print(f"📊 BEFORE fixes:")
print(f"   tok.truncation_side: {getattr(tok, 'truncation_side', 'NOT_SET')}")
print(f"   tok.model_max_length: {getattr(tok, 'model_max_length', 'NOT_SET')}")
print(f"   tok.padding_side: {getattr(tok, 'padding_side', 'NOT_SET')}")

# CRITICAL FIX: Prevent left-truncation that chops instruction headers
tok.truncation_side = "right"  # Keep instruction header, truncate table tail if needed
# tok.model_max_length = 32768   # Qwen2.5-7B supports long context

print(f"📊 AFTER fixes:")
print(f"   tok.truncation_side: {tok.truncation_side}")
print(f"   tok.model_max_length: {tok.model_max_length}")
print(f"   tok.padding_side: {tok.padding_side}")

# Use device_map="auto" to shard model across both visible GPUs
base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",  # Shard across the visible GPUs
    max_memory=MAX_MEMORY,  # Fit comfortably on 80GB cards
)

# Memory optimization settings
base_model.config.use_cache = False
base_model.gradient_checkpointing_enable()

# Show the sharding so we know both GPUs are used
try:
    hf_map = getattr(base_model, "pretrained_model", base_model).hf_device_map
    print("🔧 HF Device Map:", hf_map)
except Exception:
    print("🔧 Device map not available")

if not hasattr(base_model, "add_model_tags"):
    import types

    base_model.add_model_tags = types.MethodType(lambda self, tags: None, base_model)

# Wrap the model to ensure it always returns ModelOutput objects
model = DictOutputWrapper(base_model)

# Sanity checks before constructing the trainer
print("Running sanity checks on wrapped model...")
model.warnings_issued["estimate_tokens"] = True  # should not crash
assert model.config is not None, "Model config is None"
print("✅ warnings_issued test passed")
print("✅ config assertion passed")

# ---- Generation smoke test (model-parallel safe) ----
print("Testing generation functionality with model sharding...")
embed_device = model.get_input_embeddings().weight.device
print(f"Model embedding device: {embed_device}")
test_input = tok("hello", return_tensors="pt").to(embed_device)
_ = model.generate(**test_input)
print("✅ generation test passed")

# ---- Output format test (no compute / no device mismatch) ----
print("Testing model output format (no forward pass)...")
print("Model first param device:", next(model.parameters()).device)
print("✅ Model structure looks correct!")

# keep v_head on the same device as lm_head
try:
    lm_dev = getattr(base_model.pretrained_model, "lm_head").weight.device
    base_model.v_head.to(lm_dev)
    print("v_head device:", base_model.v_head.weight.device)
except Exception as e:
    print("v_head device check skipped:", e)

# 2. Env client (assumes server is already running on :8000)
env = LABHTTPEnv(port=8000)


# -------- Helper to pull the observation from server output (same as eval) --------
def extract_observation(resp):
    """Extract observation text from response - using eval-style extractor"""
    session = None
    if isinstance(resp, dict):
        if "session" in resp:
            session = resp["session"]
        elif "info" in resp and "session" in resp["info"]:
            session = resp["info"]["session"]

    if session and "chat_history" in session:
        hist = session["chat_history"]["value"]
        conversation = []
        for msg in hist:
            role = "User" if msg["role"] == "user" else "Agent"
            conversation.append(f"{role}: {msg['content']}")
        if conversation:
            return "\n\n".join(conversation)
    return resp.get("observation", "")


def get_turn_idx_from_session(session):
    try:
        hist = session.get("chat_history", {}).get("value", [])
    except Exception:
        return 0
    # find the last user message
    last_user_idx = -1
    for i in range(len(hist) - 1, -1, -1):
        if hist[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx == -1:
        return 0
    # count agent messages after the last user
    return sum(1 for m in hist[last_user_idx + 1 :] if m.get("role") != "user")


def derive_turn_before():
    """Resilient turn calculation that falls back to dataset.resp if snapshot lacks session."""
    snap = env.snapshot()
    sess = snap.get("session")
    if not sess:
        # fall back to the latest known response
        if isinstance(dataset.resp, dict):
            sess = dataset.resp.get("session") or dataset.resp.get("info", {}).get(
                "session"
            )
    return get_turn_idx_from_session(sess or {})


import re


def is_valid_action(txt: str) -> bool:
    """Check if text contains proper Action: (Operation|Answer) format."""
    if not txt:
        return False
    # Check if it has Action: Operation or Action: Answer anywhere in the text
    has_action = bool(re.search(r"Action:\s*(Operation|Answer)", txt))
    return has_action


def first_action_block(txt: str) -> str:
    """Extract only the first Action: Operation or Action: Answer block."""
    if not txt:
        return txt

    # Find the first Action line
    lines = txt.splitlines()
    action_start = -1
    for i, line in enumerate(lines):
        if re.match(r"^\s*Action:\s*(Operation|Answer)", line):
            action_start = i
            break

    if action_start == -1:
        return txt  # No action found

    # Determine the type of first action
    first_line = lines[action_start]
    if "Action: Operation" in first_line:
        # For Operation, extract until we hit ```sql ... ``` block
        result_lines = [first_line]
        sql_started = False
        for i in range(action_start + 1, len(lines)):
            line = lines[i]
            result_lines.append(line)

            if line.strip().startswith("```sql"):
                sql_started = True
            elif line.strip() == "```" and sql_started:
                # End of SQL block - we have complete Operation
                break
            elif line.strip().startswith("Action:"):
                # Hit another action - stop before it
                result_lines.pop()  # Remove the new action line
                break
        return "\n".join(result_lines)

    elif "Action: Answer" in first_line:
        # For Answer, extract only Action: Answer line plus Final Answer line (strict)
        result_lines = [first_line]
        for i in range(action_start + 1, len(lines)):
            line = lines[i]
            if line.strip().startswith("Final Answer:"):
                result_lines.append(line)
                break
            if line.strip().startswith("Action:"):
                # Hit next action before Final Answer
                break
        return "\n".join(result_lines)

    return txt


def trim_to_first_action(txt: str) -> str:
    """Legacy function - use first_action_block instead."""
    return first_action_block(txt)


def sanitize(s: str) -> str:
    """Clean up completion text to reduce parsing failures."""
    # Trim leading/trailing space and ensure code fences are on their own lines
    t = s.strip()
    t = t.replace("``` sql", "```sql")
    t = t.replace("```sql ", "```sql\n")
    return t


def is_operation_action(txt: str) -> bool:
    """Check if text contains Action: Operation."""
    return bool(re.search(r"Action:\s*Operation", txt))


def is_answer_action(txt: str) -> bool:
    """Check if text contains Action: Answer."""
    return bool(re.search(r"Action:\s*Answer", txt))


def select_best_idx(cands, cand_rewards, prefer_operation=True):
    """
    Select best candidate index with smart tie-breaking.
    When rewards tie, prefer Operation over Answer (unless Answer strictly wins).
    This mirrors eval's natural preference for continuing the conversation.
    """
    max_reward = max(cand_rewards)
    tied_indices = [k for k, r in enumerate(cand_rewards) if r == max_reward]

    if len(tied_indices) == 1:
        return tied_indices[0]

    # Multiple candidates tie - apply smart selection
    if prefer_operation:
        # Look for Operation actions among tied candidates
        operation_tied = [k for k in tied_indices if is_operation_action(cands[k])]
        if operation_tied:
            return operation_tied[0]  # Prefer first Operation among tied

        # No Operation found, look for any valid action
        valid_tied = [k for k in tied_indices if is_valid_action(cands[k])]
        if valid_tied:
            return valid_tied[0]
    else:
        # Standard selection - any valid action
        valid_tied = [k for k in tied_indices if is_valid_action(cands[k])]
        if valid_tied:
            return valid_tied[0]

    # Fall back to first tied candidate
    return tied_indices[0]


# OLD FUNCTION REMOVED - was sending wrong payload format

# 3. Training config
# Keep num_generations >= 2 for GRPO, score candidates with simulate(), then commit best with step()
cfg = GRPOConfig(
    output_dir="grpo_logs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=BATCH_SIZE,
    learning_rate=5e-6,
    top_entropy_quantile=ENTROPY_PCTL,
    num_train_epochs=5,
    save_strategy="no",  # Disable automatic saving during training
    logging_steps=1,
    num_generations=BATCH_SIZE,  # GRPO needs >= 2
    generation_kwargs={
        "max_new_tokens": MAX_TOKENS,
        "do_sample": True,
        "temperature": 1.0,  # Match eval temperature for consistent behavior
        "pad_token_id": tok.eos_token_id,
        "use_cache": True,  # harmless; wrapper enforces it during generate
    },
)

# Log GRPO config defaults before applying fixes
print(f"📊 GRPO Config BEFORE fixes:")
if hasattr(cfg, "max_prompt_length"):
    print(f"   cfg.max_prompt_length: {getattr(cfg, 'max_prompt_length', 'NOT_SET')}")
else:
    print(f"   cfg.max_prompt_length: ATTRIBUTE_NOT_AVAILABLE")

if hasattr(cfg, "max_completion_length"):
    print(
        f"   cfg.max_completion_length: {getattr(cfg, 'max_completion_length', 'NOT_SET')}"
    )
else:
    print(f"   cfg.max_completion_length: ATTRIBUTE_NOT_AVAILABLE")

if hasattr(cfg, "truncate_prompt_from"):
    print(
        f"   cfg.truncate_prompt_from: {getattr(cfg, 'truncate_prompt_from', 'NOT_SET')}"
    )
else:
    print(f"   cfg.truncate_prompt_from: ATTRIBUTE_NOT_AVAILABLE")

# Set TRL-specific prompt handling (if available in this TRL version)
if hasattr(cfg, "max_prompt_length"):
    cfg.max_prompt_length = 16384  # Large enough for full prompts
    print(f"✅ Set max_prompt_length = {cfg.max_prompt_length}")
if hasattr(cfg, "max_completion_length"):
    cfg.max_completion_length = MAX_TOKENS
    print(f"✅ Set max_completion_length = {cfg.max_completion_length}")
if hasattr(cfg, "truncate_prompt_from"):
    cfg.truncate_prompt_from = "right"
    print(f"✅ Set truncate_prompt_from = {cfg.truncate_prompt_from}")


# 4. Dataset - Compatible with GRPOTrainer
class EnvDataset(Dataset):
    """Dataset that provides current environment observation for GRPO training."""

    def __init__(self, env):
        self.env = env
        first_response = self.env.reset()
        self.resp = first_response
        self.obs = extract_observation(first_response)
        self.task_count = 0
        self.turn_idx = 0

        # Query environment for actual dataset size
        try:
            # Try to get the total number of tasks from the environment
            env_info = getattr(self.env, "get_info", lambda: {})()
            if isinstance(env_info, dict) and "total_tasks" in env_info:
                self.max_len = (
                    env_info["total_tasks"] * 10
                )  # Estimate ~10 turns per task
            else:
                # Fallback: try to get task count from server info
                snapshot = self.env.snapshot()
                if "total_tasks" in snapshot:
                    self.max_len = snapshot["total_tasks"] * 10
                else:
                    # Conservative fallback based on common dataset sizes
                    self.max_len = 100  # Much more reasonable default
            print(
                f"Dataset length set to {self.max_len} steps (estimated from environment)"
            )
        except Exception as e:
            print(f"Could not determine environment size, using default: {e}")
            self.max_len = 100

        print(f"Started task {self.task_count}")

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        # Don't add format hints - they're causing the model to generate incorrect formats
        # The model needs to learn the exact format from the prompt itself
        return {"prompt": self.obs}


dataset = EnvDataset(env)

# Log prompt length to verify truncation fix
prompt_tokens = tok(dataset.obs)
prompt_length = len(prompt_tokens.input_ids)
print(f"📏 Prompt analysis:")
print(f"   Token count: {prompt_length}")
print(f"   Model max_length: {tok.model_max_length}")
print(f"   Truncation side: {tok.truncation_side}")
print(f"   Max new tokens: {MAX_TOKENS}")
print(f"   Total budget needed: {prompt_length + MAX_TOKENS}")

if prompt_length > 8000:
    print(f"⚠️  Long prompt detected - truncation fixes are critical!")
else:
    print(f"✅ Prompt length reasonable")

# Create timestamped filenames for this training run
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# Group all run artifacts under grpo_logs/<model>_<timestamp>/
# Sanitize model name for filesystem safety
def _sanitize_for_dir(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


run_dir = Path(cfg.output_dir) / f"{_sanitize_for_dir(MODEL_NAME)}_{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)

training_log_file = run_dir / f"training_history_{timestamp}.jsonl"
reward_log_file = run_dir / f"reward_history_{timestamp}.json"
metrics_log_file = run_dir / f"training_metrics_history_{timestamp}.json"

# Initialize empty reward history for this run
with open(reward_log_file, "w") as f:
    json.dump([], f)

# Initialize empty training metrics history for this run
with open(metrics_log_file, "w") as f:
    json.dump([], f)

    # Write configuration header to training history
config_header = {
    "type": "config",
    "model_name": MODEL_NAME,
    "entropy_pctl": ENTROPY_PCTL,
    "batch_size": BATCH_SIZE,
    "max_tokens": MAX_TOKENS,
    "timestamp": timestamp,
    "num_generations": cfg.num_generations,
    "learning_rate": cfg.learning_rate,
}

with open(training_log_file, "w") as f:
    f.write(json.dumps(config_header, indent=2) + "\n")


def _session_from(obj: dict):
    """Extract session from various payload shapes."""
    if not isinstance(obj, dict):
        return None
    if "session" in obj and isinstance(obj["session"], dict):
        return obj["session"]
    info = obj.get("info")
    if (
        isinstance(info, dict)
        and "session" in info
        and isinstance(info["session"], dict)
    ):
        return info["session"]
    return None


def check_database_cleanup():
    """Sanity check to verify ephemeral databases are cleaned up properly."""
    try:
        resp = requests.post(env.base + "/api/cleanup_check", timeout=30)
        if resp.ok:
            result = resp.json()
            cleanup_status = result.get("cleanup_status", "unknown")
            branch_count = result.get("branch_databases_count", 0)
            print(
                f"🧹 Database cleanup check: {cleanup_status}, branch DBs remaining: {branch_count}"
            )
            return branch_count == 0
        else:
            print(f"⚠️ Cleanup check failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Cleanup check error: {e}")
        return False


def smoke_branch(env):
    """Test the branch_complete API before training."""
    snap = env.snapshot()
    sess = _session_from(snap) or _session_from(dataset.resp)
    assert (
        sess is not None
    ), f"snapshot had no session shape: keys={list(snap.keys() if snap else [])}"
    payload = {"session": sess, "candidate": "Action: Answer\nFinal Answer: []"}
    r = requests.post(env.base + "/api/branch/complete", json=payload, timeout=60)
    print("branch_complete HTTP:", r.status_code)
    print("branch_complete body:", r.text[:400])
    try:
        r.raise_for_status()
        print("json keys:", list(r.json().keys()))
        print(
            "evaluation_record:", r.json().get("session", {}).get("evaluation_record")
        )
        return True
    except Exception as e:
        print(f"API test failed: {e}")
        return False


# Global prompt tracking
current_prompt = dataset.obs
prompt_idx = 0

# Global rewind tracking
REWIND_REQUESTED = False
REWIND_INDEX = 0


def lab_reward_function(completions, prompts, **kw):
    """
    For each prompt group:
      1) snapshot current session
      2) for each candidate: simulate(action, snapshot) to score without committing
      3) return all candidate rewards to GRPO
      4) commit the best candidate with env.step() to advance the live session
         and, if done, call env.complete() and then env.reset()
    """
    global current_prompt, prompt_idx

    print(f"🔄 REWARD FUNCTION CALLED with {len(completions)} completions")

    n_gen = cfg.num_generations
    rewards = []

    for i in range(0, len(completions), n_gen):
        cands = completions[i : i + n_gen]
        prompt_text = prompts[i // n_gen]

        if prompt_text != current_prompt:
            prompt_idx += 1
            current_prompt = prompt_text

        base_snapshot = env.snapshot()

        # derive current indices
        current_task_idx = dataset.task_count
        # Use both session-derived and local turn indices like eval
        turn_before_session = get_turn_idx_from_session(
            base_snapshot.get("session", {})
        )
        turn_before_local = dataset.turn_idx  # starts at 0 on reset

        # Sanitize and trim all candidates BEFORE simulate
        cands = [trim_to_first_action(sanitize(c)) for c in cands]
        valid_frac = sum(is_valid_action(c) for c in cands) / len(cands)
        print(f"Valid Action fraction this group: {valid_frac:.2f}")

        # Get base session for branch evaluation
        base_sess = _session_from(base_snapshot) or _session_from(dataset.resp)

        # Last-resort: force a reset to get a session if we somehow have none
        if base_sess is None:
            try:
                base_sess = _session_from(env.reset())
            except Exception:
                pass

        # Use branch/complete API to get true rewards for all candidates
        cand_rewards = []
        for j, cand in enumerate(cands):
            # Get detailed branch evaluation with outcome
            try:
                payload = {"session": copy.deepcopy(base_sess), "candidate": cand}
                print(
                    f"POST /branch/complete with keys: {list(payload.keys())}"
                )  # Debug logging
                resp = requests.post(
                    env.base + "/api/branch/complete", json=payload, timeout=120
                )
                if resp.ok:
                    branch_data = resp.json()
                    sess = branch_data.get("session", {})
                    rec = sess.get("evaluation_record", {})
                    outcome = rec.get("outcome", "unknown")
                    branch_reward = 1.0 if outcome == "correct" else 0.0
                    branch_done = branch_data.get("done", False)
                else:
                    outcome = "api_error"
                    branch_reward = 0.0
                    branch_done = False
            except Exception as e:
                outcome = f"exception_{str(e)[:20]}"
                branch_reward = 0.0
                branch_done = False

            r = branch_reward  # Don't rescale the reward
            cand_rewards.append(r)

            with open(training_log_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "branch_evaluation",
                            "task_idx": current_task_idx,
                            "turn_idx_before_session": turn_before_session,
                            "turn_idx_before_local": turn_before_local,
                            "prompt_idx": prompt_idx,
                            "completion_idx": j,
                            "prompt": prompt_text,
                            "completion": cand,
                            "branch_reward": branch_reward,
                            "scaled_reward": r,
                            "branch_outcome": outcome,  # NEW: Log the actual evaluation outcome
                            "done": branch_done,  # NEW: Add done field for troubleshooting
                            "is_valid_action": is_valid_action(cand),
                        },
                        indent=2,
                    )
                    + "\n"
                )

        # NEW MULTI-TURN STRATEGY: Use branch scores for ranking only, not termination
        # Select best candidate with smart tie-breaking (prefer Operation over Answer)
        # Episodes are ongoing until server says done=True
        best_idx = select_best_idx(cands, cand_rewards, prefer_operation=True)

        # Extract only the first action block (mirrors eval behavior)
        best_cand = first_action_block(sanitize(cands[best_idx]))

        # ALWAYS commit the selected candidate (never terminate based on branch scores)
        commit_out = env.step(best_cand)

        # Only override with terminal reward if episode actually completes
        final_outcome = None
        if commit_out["done"]:
            try:
                complete_result = env.complete()
                session = complete_result["session"]
                final_outcome = session["evaluation_record"]["outcome"]
                final_reward = 1 if final_outcome == "correct" else 0
                # Override ONLY the committed candidate's reward with terminal outcome
                cand_rewards[best_idx] = final_reward
                print(
                    f"🏁 Episode complete! Terminal outcome: {final_outcome}, reward: {final_reward}"
                )
            except Exception as e:
                print(f"⚠️ Error getting terminal reward: {e}")
                final_outcome = "error"
        else:
            print(
                f"🔄 Episode continues... Turn {turn_before_session} → {turn_before_session + 1}"
            )

        # Calculate turn indices after the step for comparison
        commit_session = commit_out.get("info", {}).get("session", {})
        commit_sample_status = commit_session.get("sample_status", "unknown")
        commit_finish_reason = commit_session.get("finish_reason", "")
        turn_after_session = get_turn_idx_from_session(commit_session)

        # Optional: cross-check with fresh snapshot after step
        try:
            snap_after = env.snapshot()
            turn_after_snapshot = get_turn_idx_from_session(
                snap_after.get("session", {})
            )
        except Exception:
            turn_after_snapshot = None

        with open(training_log_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "type": "commit",
                        "task_idx": current_task_idx,
                        "turn_idx_before_session": turn_before_session,
                        "turn_idx_before_local": turn_before_local,
                        "turn_idx_after_session": turn_after_session,  # NEW: after-step turn from commit_out
                        "turn_idx_after_snapshot": turn_after_snapshot,  # NEW: optional cross-check
                        "prompt_idx": prompt_idx,
                        "best_completion_idx": best_idx,
                        "committed_completion": best_cand,
                        "done": commit_out[
                            "done"
                        ],  # NEW: Add done field for troubleshooting
                        "commit_done": commit_out["done"],
                        "commit_step_reward": commit_out.get("reward", 0),
                        "commit_sample_status": commit_sample_status,  # Track parsing status
                        "commit_finish_reason": commit_finish_reason,
                        "final_outcome": final_outcome,
                        "best_candidate_reward": cand_rewards[
                            best_idx
                        ],  # Debug selection
                        "selected_valid_action": is_valid_action(best_cand),
                        "selected_operation_action": is_operation_action(best_cand),
                        "selected_answer_action": is_answer_action(best_cand),
                    },
                    indent=2,
                )
                + "\n"
            )

        rewards.extend(cand_rewards)

        # Periodic cleanup check (every 10 tasks)
        if current_task_idx % 10 == 0:
            check_database_cleanup()

        # --- Rewind-or-Advance logic (safe point) ---
        if commit_out["done"]:
            global REWIND_REQUESTED, REWIND_INDEX
            try:
                if REWIND_REQUESTED:
                    # Start the next epoch's pass from a known index (e.g., 0)
                    env.current_sample_index = REWIND_INDEX
                    next_resp = env.reset()
                    print(
                        f"[Rewind] Reset to task {REWIND_INDEX}; next will be {env.current_sample_index}"
                    )
                    dataset.task_count = 0
                else:
                    # Normal sequential advance
                    next_resp = env.reset()
                    dataset.task_count += 1
                dataset.resp = next_resp
                dataset.obs = extract_observation(next_resp)
                dataset.turn_idx = 0
            except requests.exceptions.JSONDecodeError:
                pass
            finally:
                REWIND_REQUESTED = False
        else:
            dataset.resp = commit_out
            dataset.obs = extract_observation(commit_out)
            dataset.turn_idx = turn_after_session  # stay in sync with server

        # keep your reward history write here
        try:
            with open(reward_log_file, "r") as f:
                reward_list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            reward_list = []
        reward_list.append(cand_rewards)
        with open(reward_log_file, "w") as f:
            json.dump(reward_list, f, indent=2)

    print(f"✅ REWARD FUNCTION COMPLETE - returning {len(rewards)} rewards")
    return rewards


# Custom trainer subclass that preserves model sharding
class GRPOTrainerNoMove(GRPOTrainer):
    # Keep the sharded model exactly where device_map put it
    def _move_model_to_device(self, model, device):
        return model


# Help some Trainer heuristics recognize model-parallel
setattr(model, "is_parallelizable", True)
setattr(model, "model_parallel", True)

trainer = GRPOTrainerNoMove(
    model=model,
    reward_funcs=[lab_reward_function],
    train_dataset=dataset,
    args=cfg,
)
trainer.processing_class = tok
trainer.tokenizer = tok  # add this line for TRL compatibility

# Custom callback to log training metrics
from transformers import TrainerCallback


class MetricsLoggingCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            try:
                with open(self.log_file, "r") as f:
                    metrics_list = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                metrics_list = []

            metrics_entry = {
                "step": state.global_step,
                "epoch": logs.get("epoch", 0),
                **logs,
            }

            metrics_list.append(metrics_entry)

            with open(self.log_file, "w") as f:
                json.dump(metrics_list, f, indent=2)


# Add the callback
metrics_callback = MetricsLoggingCallback(metrics_log_file)
trainer.add_callback(metrics_callback)


class EpochResetCallback(TrainerCallback):
    def __init__(self, start_index: int = 0):
        self._last_epoch_reset = -1
        self._restart_index = start_index

    def on_epoch_begin(self, args, state, control, **kwargs):
        global REWIND_REQUESTED, REWIND_INDEX
        if state.epoch is None:
            return
        epoch_i = int(state.epoch)
        if epoch_i == 0 or epoch_i == self._last_epoch_reset:
            return

        self._last_epoch_reset = epoch_i
        REWIND_REQUESTED = True
        REWIND_INDEX = self._restart_index
        print(
            f"[EpochReset] Will rewind to sample_index={REWIND_INDEX} after current episode completes."
        )


trainer.add_callback(EpochResetCallback())

# Test the fixes before training (model-parallel safe)
print("Testing model output format...")
with torch.no_grad():
    embed_device = trainer.model.get_input_embeddings().weight.device
    sample = tok("test", return_tensors="pt").to(embed_device)
    out = trainer.model(**sample)  # use the wrapper the trainer sees
    print(f"Output type: {type(out)}, has logits: {hasattr(out, 'logits')}")
    if hasattr(out, "logits"):
        print("✅ Model output format is correct!")
    else:
        print("❌ Model output format is still incorrect!")

# Model Sharding Information
print(f"🚀 Training setup: Single process with model sharding across GPUs 6,7")
print(f"� Memory budget: 38GiB per GPU with gradient checkpointing enabled")

# print("Testing branch_complete API...")
# try:
#     if smoke_branch(env):
#         print("✅ branch_complete API is working!")
#     else:
#         print("❌ branch_complete API failed!")
# except Exception as e:
#     print(f"❌ branch_complete API test failed: {e}")

# Train
print("Starting GRPO training...")
trainer.train()
print("Training completed!")

# # Save the final model
# print("Saving final model...")
# trainer.save_model()
# print("Final model saved!")

# --- robust saving without safe_save_model_for_hf_trainer ---
print("Saving final model with robust routine...")
from pathlib import Path

# import torch, json # already imported

model_output_dir = "trained_model"
save_dir = Path(model_output_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# unwrap accelerate + your DictOutputWrapper
m = trainer.accelerator.unwrap_model(trainer.model)
if hasattr(m, "base_model"):  # DictOutputWrapper
    m = m.base_model  # -> TRL AutoModelForCausalLMWithValueHead

# get handles
base_lm = getattr(m, "pretrained_model", m)  # underlying HF CausalLM
v_head = getattr(m, "v_head", None)

# gather state dict for the *base LM only* (avoids TRL state_dict bug)
base_sd = trainer.accelerator.get_state_dict(base_lm)

# main process writes files
if trainer.accelerator.is_main_process:
    # save the base LM in standard HF format (works across versions)
    base_lm.save_pretrained(save_dir, state_dict=base_sd, safe_serialization=True)
    tok.save_pretrained(save_dir)

    # save value head separately
    if v_head is not None:
        vh_path = save_dir / "v_head.bin"
        torch.save({k: v.cpu() for k, v in v_head.state_dict().items()}, vh_path)
        with open(save_dir / "trl_extra.json", "w") as f:
            json.dump({"value_head": "v_head.bin"}, f)

trainer.accelerator.wait_for_everyone()
print("Saved base LM and v_head separately to", str(save_dir))
