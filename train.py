import pathlib, sys

# add src/ so we can import YAML loader utils if needed later
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "src"))

import os, json, torch, requests
from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from lab_http_env import LABHTTPEnv  # ← our tiny client
from pathlib import Path
from datasets import load_dataset


def download_and_setup_dataset():
    print("Downloading LifelongAgentBench – database only …")
    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/csyq/LifelongAgentBench@main/database/train-*.parquet",
        split="train",
        streaming=True,
    )  # ← don’t build Arrow cache

    # Preview a couple of rows
    for i, ex in enumerate(ds.take(3)):
        print(ex)


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ENTROPY_PCTL = 0.8
BATCH_SIZE = 4
MAX_TOKENS = 192

# Download and setup dataset before training
download_and_setup_dataset()

# 1. Model
tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)

# 2. Env client (assumes server is already running on :8001)
env = LABHTTPEnv(port=8001)

# 3. Training config
cfg = GRPOConfig(
    batch_size=BATCH_SIZE,
    learning_rate=5e-6,
    token_entropy_percentile_threshold=ENTROPY_PCTL,
)

trainer = GRPOTrainer(config=cfg, model=model, ref_model=model, tokenizer=tok)

# 4. Main loop ----------------------------------------------------------
obs = env.reset()["observation"]  # first task description
while True:
    batch_prompts = [obs]
    gen_ids = trainer.generate(batch_prompts, max_new_tokens=MAX_TOKENS)
    gen_txt = tok.batch_decode(gen_ids, skip_special_tokens=True)

    step_out = env.step(gen_txt[0])  # send action
    reward = 2 * step_out["reward"] - 1
    trainer.step(batch_prompts, gen_txt, rewards=[reward])

    print(json.dumps({"prompt": obs[:80], "answer": gen_txt[0][:80], "reward": reward}))

    if step_out["done"]:
        try:
            obs = env.reset()["observation"]  # next task
        except requests.exceptions.JSONDecodeError:
            print("All tasks finished.")
            break
    else:
        obs = step_out["observation"]
# ----------------------------------------------------------------------
