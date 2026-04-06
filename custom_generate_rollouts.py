#!/usr/bin/env python3
"""
Generate rollouts for the Thought Anchor pilot experiment (math + logic).

This script reproduces the rollout data in pilot_results/rollouts/.
Run from the pilot_results/ directory with a GPU and vLLM installed:

    cd pilot_results/
    pip install vllm==0.16.0
    python custom_generate_rollouts.py

Output:
    rollouts/{domain}/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/
      {correct|incorrect}_base_solution/problem_0/
        problem.json, base_solution.json, chunks.json, chunk_N/solutions.json
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List

# ============================================================
# Configuration
# ============================================================
NUM_ROLLOUTS = 20
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 16384
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_SHORT = "deepseek-r1-distill-qwen-14b"

SCRIPT_DIR = Path(__file__).resolve().parent
ROLLOUTS_DIR = SCRIPT_DIR / "rollouts"


# ============================================================
# Sentence splitting (from thought-anchors/utils.py:360-447)
# ============================================================
def split_solution_into_chunks(solution_text: str) -> List[str]:
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]
    chunks = []
    current_chunk = ""
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]
        is_paragraph_end = any(
            i + len(p) <= len(solution_text) and solution_text[i : i + len(p)] == p
            for p in paragraph_ending_patterns
        )
        is_sentence_end = (
            i < len(solution_text) - 1
            and solution_text[i] in sentence_ending_tokens
            and solution_text[i + 1] in (" ", "\n")
        )
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
        i += 1
    # Merge small chunks (<10 characters)
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] += " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1
    return chunks


# ============================================================
# Answer extraction & checking
# ============================================================
def extract_boxed_answers(text: str) -> List[str]:
    starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return [""]
    answers = []
    for s in starts:
        idx, depth, ans = s + 7, 1, ""
        while idx < len(text) and depth > 0:
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
            if depth > 0:
                ans += text[idx]
            idx += 1
        if ans:
            answers.append(ans)
    return answers or [""]


def normalize_latex(s: str) -> str:
    s = s.strip().lower().replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\s+", "", s).replace("\\%", "").replace("{,}", "")
    s = s.replace("\\times", "*").replace("\\cdot", "*")
    s = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", s)
    s = re.sub(r"{([^{}]+)}", r"\1", s).replace("\\pi", "pi")
    s = re.sub(r"\\text\{([^{}]+)\}", r"\1", s).replace("\\text", "")
    return s


def check_math_answer(answer: str, gt: str) -> bool:
    if normalize_latex(answer) == normalize_latex(gt):
        return True
    try:
        return abs(float(answer) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        return False


def extract_logic_answer(text: str) -> str:
    if "</think>" in text:
        after = text.split("</think>")[-1].lower()
        if "true" in after:
            return "True"
        if "false" in after:
            return "False"
    for line in reversed(text.strip().split("\n")[-5:]):
        ll = line.strip().lower()
        if ll == "true" or "the answer is true" in ll or "the conclusion is true" in ll:
            return "True"
        if ll == "false" or "the answer is false" in ll or "the conclusion is false" in ll:
            return "False"
    return ""


def extract_answer(text: str, domain: str) -> str:
    if domain == "math":
        return (extract_boxed_answers(text) or [""])[0]
    if domain == "logic":
        return extract_logic_answer(text)
    return ""


def check_answer(answer: str, gt_answer: str, domain: str) -> bool:
    if not answer:
        return False
    if domain == "math":
        return check_math_answer(answer, gt_answer)
    if domain == "logic":
        return answer.strip().lower() == gt_answer.strip().lower()
    return False


# ============================================================
# Prompt builder
# ============================================================
def build_base_prompt(problem: dict, domain: str) -> str:
    if domain == "math":
        return (
            f"Solve this math problem step by step. "
            f"You MUST put your final answer in \\boxed{{}}. "
            f"Problem: {problem['problem']} Solution: \n<think>\n"
        )
    if domain == "logic":
        ctx = problem.get("context", "")
        q = problem.get("question", "")
        if ctx and q:
            return (
                f"Answer the following logical reasoning question step by step.\n\n"
                f"Premises:\n{ctx}\n\n"
                f"Based on the premises above, is the following conclusion true or false?\n"
                f"Conclusion: {q}\n\n"
                f"Answer:\n<think>\n"
            )
        return (
            f"Answer the following logical reasoning question step by step.\n\n"
            f"{problem['problem']}\n\n"
            f"Answer:\n<think>\n"
        )
    raise ValueError(f"Unknown domain: {domain}")


# ============================================================
# Rollout generation
# ============================================================
def run_rollouts(domain: str, cot_type: str, llm, sampling_params):
    """Generate rollouts for one (domain, cot_type) pair."""
    prob = json.load(open(SCRIPT_DIR / domain / "problem.json"))
    cot = json.load(open(SCRIPT_DIR / domain / f"{cot_type}_cot.json"))

    out = (
        ROLLOUTS_DIR / domain / MODEL_SHORT
        / f"temperature_{TEMPERATURE}_top_p_{TOP_P}"
        / f"{cot_type}_base_solution" / "problem_0"
    )
    out.mkdir(parents=True, exist_ok=True)

    base_prompt = build_base_prompt(prob, domain)
    sol_raw = cot["solution"]
    full_cot = base_prompt + sol_raw

    # Extract thinking text
    sol_text = sol_raw
    if "<think>" in sol_text:
        sol_text = sol_text.split("<think>")[1].strip()
    if "</think>" in sol_text:
        sol_text = sol_text.split("</think>")[0].strip()

    # Split into chunks (let the function handle <think>/<think>)
    chunks = split_solution_into_chunks(sol_raw)

    # Save metadata
    json.dump(
        {"problem": prob["problem"], "gt_answer": prob.get("gt_answer", ""),
         "level": prob.get("level", ""), "type": prob.get("type", domain)},
        open(out / "problem.json", "w"), indent=2,
    )
    json.dump(
        {"prompt": base_prompt, "solution": sol_raw, "full_cot": full_cot,
         "answer": cot.get("answer", ""),
         "is_correct": cot.get("is_correct", cot_type == "correct")},
        open(out / "base_solution.json", "w"), indent=2,
    )
    json.dump(
        {"source_text": full_cot, "solution_text": sol_text, "chunks": chunks},
        open(out / "chunks.json", "w"), indent=2,
    )

    # Build cumulative prefixes
    cumulative, cur = [], ""
    for c in chunks:
        cur += c + " "
        cumulative.append(cur.strip())

    # Collect all prompts (skip already-completed chunks)
    all_prompts, idx_map = [], []
    for ci, (chunk, prefix) in enumerate(zip(chunks, cumulative)):
        chunk_dir = out / f"chunk_{ci}"
        chunk_dir.mkdir(exist_ok=True)
        sf = chunk_dir / "solutions.json"
        if sf.exists() and len(json.load(open(sf))) >= NUM_ROLLOUTS:
            continue
        pw = prefix.replace(chunk, "").strip()
        rp = base_prompt + pw
        for _ in range(NUM_ROLLOUTS):
            all_prompts.append(rp)
            idx_map.append((ci, rp, chunk, pw))

    print(f"  [{domain}/{cot_type}] {len(chunks)} chunks, {len(all_prompts)} prompts")
    if not all_prompts:
        print(f"  Already done, skipping.")
        return

    # Submit all prompts in one batch
    t0 = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  Batch done in {elapsed / 60:.1f} min")

    # Group outputs by chunk and save
    by_chunk = {}
    for o, (ci, rp, chunk, pw) in zip(outputs, idx_map):
        rtxt = o.outputs[0].text
        rc = split_solution_into_chunks(rtxt)
        ans = extract_answer(rtxt, domain)
        ok = check_answer(ans, prob.get("gt_answer", ""), domain)
        by_chunk.setdefault(ci, []).append({
            "chunk_removed": chunk,
            "prefix_without_chunk": pw,
            "chunk_resampled": rc[0] if rc else "",
            "rollout": rtxt,
            "full_cot": rp + rtxt,
            "answer": ans,
            "is_correct": ok,
        })

    for ci, sols in sorted(by_chunk.items()):
        json.dump(sols, open(out / f"chunk_{ci}" / "solutions.json", "w"), indent=2)

    nc = sum(s["is_correct"] for ss in by_chunk.values() for s in ss)
    nt = sum(len(ss) for ss in by_chunk.values())
    print(f"  [{domain}/{cot_type}] {len(by_chunk)} chunks, {nt} rollouts, {nc} correct ({nc / nt * 100:.1f}%)")


# ============================================================
# Main
# ============================================================
def main():
    from vllm import LLM, SamplingParams

    print(f"Model: {MODEL_NAME}")
    print(f"Rollouts per chunk: {NUM_ROLLOUTS}")
    print(f"Output: {ROLLOUTS_DIR}")
    print()

    print("Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        max_model_len=MAX_TOKENS,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    )
    sp = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)
    print("Model loaded.\n")

    for domain in ["math", "logic"]:
        for cot_type in ["correct", "incorrect"]:
            print(f"\n{'─'*50}")
            print(f"  {domain} / {cot_type}")
            print(f"{'─'*50}")
            run_rollouts(domain, cot_type, llm, sp)

    # Verify
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    for domain in ["math", "logic"]:
        for cot_type in ["correct", "incorrect"]:
            p = (
                ROLLOUTS_DIR / domain / MODEL_SHORT
                / f"temperature_{TEMPERATURE}_top_p_{TOP_P}"
                / f"{cot_type}_base_solution" / "problem_0"
            )
            if not (p / "chunks.json").exists():
                print(f"  [{domain}/{cot_type}] MISSING")
                continue
            nc = len(json.load(open(p / "chunks.json"))["chunks"])
            done = sum(1 for ci in range(nc) if (p / f"chunk_{ci}" / "solutions.json").exists())
            total = sum(
                len(json.load(open(p / f"chunk_{ci}" / "solutions.json")))
                for ci in range(nc)
                if (p / f"chunk_{ci}" / "solutions.json").exists()
            )
            print(f"  [{domain}/{cot_type}] {done}/{nc} chunks, {total} rollouts")


if __name__ == "__main__":
    main()
