#!/usr/bin/env python3
"""
Offline evaluation script for the LABHTTPEnv.
The script:

1. Finds an idle GPU and pins it through CUDA_VISIBLE_DEVICES.
2. Loads a causal‑LM with a value head (same checkpoint used for training).
3. Repeatedly
   • pulls a new task from the environment,
   • generates one completion,
   • sends it back to the server,
   • collects the scalar reward.
4. Prints per‑task and average rewards to the console and logs them to
   grpo_logs/eval_history_*.jsonl for later analysis.

No weight updates occur; this is inference only.
"""

# --------‑‑‑ Install and import 3rd‑party deps ‑‑‑---------
import os, sys, time, json, argparse, pathlib
from datetime import datetime
import torch
import GPUtil
from transformers import AutoTokenizer, AutoModelForCausalLM
from lab_http_env import LABHTTPEnv

# --------‑‑‑ GPU selection util ‑‑‑---------
import GPUtil, subprocess, time, os


def wait_for_free_gpu(max_load=0.3, max_mem=0.3, poll=10):
    """Return the id of a GPU that is idle AND reports zero uncorrectable ECC errors."""
    while True:
        candidates = GPUtil.getAvailable(
            order="memory",
            limit=8,  # look at up to 8 devices
            maxLoad=max_load,
            maxMemory=max_mem,
        )
        healthy = []
        for idx in candidates:
            try:
                ecc = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        str(idx),
                        "--query-gpu=ecc.errors.uncorrectable.volatile.total",
                        "--format=csv,noheader,nounits",
                    ],
                    encoding="utf-8",
                ).strip()
                if ecc == "" or int(ecc) == 0:
                    healthy.append(idx)
            except Exception:
                # nvidia-smi may fail if the driver is in a bad state
                continue
        if healthy:
            gpu_id = str(healthy[0])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            return gpu_id
        print("No healthy GPU free; retrying in", poll, "s")
        time.sleep(poll)


# --------‑‑‑ Helper to pull the instruction from server output ‑‑‑---------
def extract_observation(resp):
    session = None
    if isinstance(resp, dict):
        if "session" in resp:
            session = resp["session"]  # env.reset() case
        elif "info" in resp and "session" in resp["info"]:
            session = resp["info"]["session"]  # env.step() case

    if session and "chat_history" in session:
        hist = session["chat_history"]["value"]
        conversation = []
        for msg in hist:
            role = "User" if msg["role"] == "user" else "Agent"
            conversation.append(f"{role}: {msg['content']}")
        if conversation:
            return "\n\n".join(conversation)  # Full conversation history

    return resp.get("observation", "")


# --------‑‑‑ Helper to update summary at top of file ‑‑‑---------
def update_summary(
    history_file,
    total_tasks,
    total_completed_tasks,
    successful_tasks,
    total_reward,
    model,
    timestamp,
    temperature,
    max_new_tokens,
):
    avg = total_reward / max(total_tasks, 1)
    success_rate = (
        (successful_tasks / max(total_completed_tasks, 1)) * 100
        if total_completed_tasks > 0
        else 0.0
    )

    summary_entry = {
        "summary": True,
        "total_tasks": total_tasks,
        "total_completed_tasks": total_completed_tasks,
        "successful_tasks": successful_tasks,
        "average_reward": avg,
        "success_rate": success_rate,
        "model": model,
        "timestamp": timestamp,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
    }

    # Read existing content and filter out all summary entries
    task_entries = []
    if history_file.exists():
        with open(history_file, "r") as fh:
            content = fh.read().strip()

        if content:
            # Split by double newlines to get individual JSON entries
            entries = content.split("\n\n")
            for entry in entries:
                entry = entry.strip()
                if entry:
                    try:
                        parsed = json.loads(entry)
                        # Only keep non-summary entries (actual task entries)
                        if not parsed.get("summary", False):
                            task_entries.append(entry)
                    except json.JSONDecodeError:
                        # If parsing fails, keep the entry as-is (might be partial)
                        task_entries.append(entry)

    # Write updated summary at top, then existing task entries
    with open(history_file, "w") as fh:
        fh.write(json.dumps(summary_entry, ensure_ascii=False, indent=2))
        fh.write("\n\n")
        for entry in task_entries:
            fh.write(entry)
            fh.write("\n\n")


# --------‑‑‑ Main ‑‑‑---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        # default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    # default="Qwen/Qwen2.5-32B-Instruct")
    # default="Qwen/QwQ-32B")
    # default="meta-llama/Llama-3.1-8B-Instruct")
    # default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    # default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_tasks", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # 1. GPU
    # gpu_id = wait_for_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device("cuda")

    # 2. Tokenizer and model (value head not needed for inference)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()

    # 3. Environment
    env = LABHTTPEnv(port=args.port)

    # 4. Logging
    out_dir = pathlib.Path("grpo_logs")
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = args.model.split("/")[-1]  # Extract model ID from path
    history_file = out_dir / f"eval_history_{model_id}_{stamp}.jsonl"

    # 5. Evaluation loop
    total_reward = 0.0
    task_idx = 0

    # Statistics tracking for successful tasks
    task_final_rewards = {}  # Maps task_idx -> (max_turn_idx, final_reward)
    successful_tasks = 0
    total_completed_tasks = 0

    while task_idx < args.num_tasks:
        resp = env.reset()  # Start new task
        turn_idx = 0

        # Inner loop: continue conversation until task is done
        while True:
            prompt = extract_observation(resp)
            if not prompt:
                print("Warning: empty prompt")
                break

            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    pad_token_id=tok.eos_token_id,
                )
            completion = tok.decode(
                gen_ids[0, inputs["input_ids"].size(1) :],
                skip_special_tokens=True,
            )

            step_out = env.step(completion)
            done = step_out["done"]

            # If task is complete, call env.complete() to calculate final outcome BEFORE getting reward
            if done:
                complete_result = env.complete()
                # Get the updated session with calculated outcome
                session = complete_result["session"]
                outcome = session["evaluation_record"]["outcome"]
                reward = 1 if outcome == "correct" else 0
                print(f"🔍 EVAL DEBUG: Final outcome = '{outcome}', reward = {reward}")
            else:
                reward = step_out["reward"]

            total_reward += reward

            # Check for parsing failures
            session = step_out["info"]["session"]
            sample_status = session.get("sample_status", "unknown")
            finish_reason = session.get("finish_reason", "")

            if sample_status == "agent_validation_failed":
                print(
                    f"Task {task_idx+1:03d}-T{turn_idx+1} | PARSING FAILED: {finish_reason}"
                )
                log_entry = {
                    "task_idx": task_idx,
                    "turn_idx": turn_idx,
                    "prompt": prompt,
                    "completion": completion,
                    "reward": reward,
                    "done": done,
                    "parsing_failed": True,
                    "failure_reason": finish_reason,
                }
            else:
                print(
                    f"Task {task_idx+1:03d}-T{turn_idx+1} | r={reward:.3f} | done={done}"
                )
                log_entry = {
                    "task_idx": task_idx,
                    "turn_idx": turn_idx,
                    "prompt": prompt,
                    "completion": completion,
                    "reward": reward,
                    "done": done,
                }

            # Update task statistics tracking
            if (
                task_idx not in task_final_rewards
                or turn_idx > task_final_rewards[task_idx][0]
            ):
                task_final_rewards[task_idx] = (turn_idx, reward)

            # Write log entry immediately
            with open(history_file, "a") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False, indent=2))
                fh.write("\n\n")

            if done:
                # Update completion statistics
                total_completed_tasks += 1
                final_reward = task_final_rewards[task_idx][1]
                if final_reward == 1:
                    successful_tasks += 1

                # Update summary in real-time after each completed task
                update_summary(
                    history_file,
                    task_idx + 1,
                    total_completed_tasks,
                    successful_tasks,
                    total_reward,
                    args.model,
                    stamp,
                    args.temperature,
                    args.max_new_tokens,
                )
                break  # Exit inner loop, move to next task
            else:
                resp = step_out  # Continue same task conversation
                turn_idx += 1

        task_idx += 1

    avg = total_reward / max(task_idx, 1)
    success_rate = (
        (successful_tasks / max(total_completed_tasks, 1)) * 100
        if total_completed_tasks > 0
        else 0.0
    )

    print(f"Finished {task_idx} tasks | average reward = {avg:.4f}")
    print(f"Task Success Statistics:")
    print(f"  - Successful tasks: {successful_tasks}/{total_completed_tasks}")
    print(f"  - Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()
