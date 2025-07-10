import time
import GPUtil


def check_free_gpus(required_gpus=1):
    """Check if the required number of GPUs are free."""
    free_gpus = GPUtil.getAvailable(
        order="memory", limit=required_gpus, maxLoad=0.1, maxMemory=0.1
    )
    return len(free_gpus) >= required_gpus, free_gpus


def wait_for_free_gpus(required_gpus=3):
    """Wait until the required number of GPUs are free."""
    print(f"Waiting for {required_gpus} free GPU(s)...")
    while True:
        available, free_gpus = check_free_gpus(required_gpus)
        if available:
            print(f"{required_gpus} GPU(s) are free: {free_gpus}")
            return free_gpus
        print("Not enough free GPUs. Checking again in 10 seconds...")
        time.sleep(10)


free_gpus = wait_for_free_gpus(2)
free_gpus = ",".join(list(map(str, free_gpus)))

import os

# Set CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus

import pathlib, sys

# add src/ so we can import YAML loader utils if needed later
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

import os, json, torch, requests
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from lab_http_env import LABHTTPEnv  # ← our tiny client
from pathlib import Path


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ENTROPY_PCTL = 0.8
BATCH_SIZE = 8
MAX_TOKENS = 1024

# No need to download dataset - server loads local data
print("Using local dataset loaded by server...")

# 1. Model
tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)

if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

if not hasattr(model, "add_model_tags"):
    import types

    model.add_model_tags = types.MethodType(lambda self, tags: None, model)

# 2. Env client (assumes server is already running on :8000)
env = LABHTTPEnv(port=8000)

# 3. Training config
cfg = GRPOConfig(
    output_dir="grpo_logs",  # required by TrainingArguments
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=5e-6,
    token_entropy_percentile_threshold=ENTROPY_PCTL,
    # (any other TrainingArguments fields you like …)
)


# Helper function to extract observation from server response
def extract_observation(response):
    """Extract the task instruction from the server response."""
    if isinstance(response, dict) and "session" in response:
        chat_history = response["session"]["chat_history"]["value"]
        # Find the last user message that contains the task instruction
        for msg in reversed(chat_history):
            if (
                msg["role"] == "user" and len(msg["content"]) > 100
            ):  # The task instruction is longer
                return msg["content"]
    return response.get("observation", "")  # fallback


trainer = GRPOTrainer(
    model=model,
    reward_funcs=[],
    # reference_model=model,
    args=cfg,  # ← use args=
)
trainer.tokenizer = tok

# 4. Main loop ----------------------------------------------------------
first_step = env.reset()
print("reset() raw response:", first_step)  # ⇐ NEW
obs = extract_observation(first_step)  # first task description

while True:
    # 1️⃣  prompts → model
    batch_prompts = [obs]  # 1 prompt
    gen_ids = trainer.generate(
        batch_prompts,
        max_new_tokens=MAX_TOKENS,
    )  # shape: (1 × num_generations, seq_len)

    # 2️⃣ decode: list[str] length == 1 × num_generations
    gen_txt = tok.batch_decode(gen_ids, skip_special_tokens=True)

    # 3️⃣ group completions per original prompt
    num_gen = cfg.num_generations  # e.g. 8
    # chunks[i] is the list of all completions for prompt i
    chunks = [gen_txt[i : i + num_gen] for i in range(0, len(gen_txt), num_gen)]

    # 4️⃣ environment interaction (one step per prompt)
    #    Here we pick *the first* completion to act with.
    #    If you want a smarter selection, replace answers[0].
    all_prompts, all_completions, all_rewards = [], [], []
    for prompt, answers in zip(batch_prompts, chunks):
        action = answers[0]  # choose policy’s top answer
        step_out = env.step(action)  # send to LAB
        reward = 2 * step_out["reward"] - 1  # rescale to [-1, 1]

        # collect data for the trainer
        all_prompts.extend([prompt] * num_gen)  # repeat prompt to match completions
        all_completions.extend(answers)  # use *all* generated texts
        all_rewards.extend([reward] * num_gen)  # broadcast reward

        # log a short preview
        print(
            json.dumps(
                {
                    "prompt": prompt[:80],
                    "answer": action[:80],
                    "reward": reward,
                }
            )
        )

        # bookkeeping for next cycle
        if step_out["done"]:
            try:
                reset_response = env.reset()
                obs = extract_observation(reset_response)  # new task
            except requests.exceptions.JSONDecodeError:
                print("All tasks finished.")
                obs = None
                break
        else:
            obs = step_out.get("observation", "")  # fallback for step response

    if obs is None:  # tasks exhausted
        break

    # 5️⃣ policy/value update
    trainer.step(all_prompts, all_completions, rewards=all_rewards)
# ----------------------------------------------------------------------
