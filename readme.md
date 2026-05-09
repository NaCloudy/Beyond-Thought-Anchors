# Beyond Thought Anchors
Beyond Thought Anchors: Cross-Domain and Cross-Method Analysis of LLM Reasoning Steps

> **CS639 Deep Learning for NLP — Course Project**
> 
> Extending *Thought Anchors: Which LLM Reasoning Steps Matter?* from math reasoning to logic reasoning.



## What is a Thought Anchor?

When a reasoning model solves a problem, it generates a long chain-of-thought (CoT) trace — hundreds of sentences of internal deliberation. The central question of the original paper is: **which of those sentences actually matter?**

### The counterintuitive finding

The naive answer is "computation sentences" (Active Computation — where the model does actual algebra or arithmetic). If you measure importance by *interrupting* the model mid-trace and forcing an immediate answer, computation steps look most important, because they're closest to the answer.

But this confounds two things: *which sentence contains the answer* vs. *which sentence caused the model to find the correct answer*.

The paper's key insight is that once a reasoning model has committed to a particular plan, the subsequent computations are **largely predetermined** — the model will reliably execute them. The real fork point is earlier: *which plan did the model adopt?*

### What actually matters: Planning and Backtracking

When importance is measured correctly — by **counterfactually resampling the entire subsequent chain** (regenerating all downstream steps, not just forcing the next token) — a different picture emerges:

- **Plan Generation sentences** (e.g., *"Let me calculate the decimal value first and convert from there"*) have the highest counterfactual importance. They set the trajectory; once this sentence appears, the model reliably executes the chosen plan and reaches a specific answer.
- **Uncertainty Management / Backtracking sentences** (e.g., *"Wait, I should double-check this"*) are the second most important. They are pivot points where the model abandons a failing approach and switches direction.
- **Active Computation sentences** drop sharply in importance under this measure. They are executing a plan, not choosing one — if you remove them, the model generates equivalent computations anyway.

These high-impact planning and backtracking sentences are the **thought anchors**: they anchor the trajectory of the entire subsequent reasoning.

### The case study that illustrates it

Problem: *"How many bits does 66666₁₆ have in base 2?"*

The model initially plans to multiply 5 hex digits × 4 bits/digit = 20 bits. This plan is wrong (it ignores leading zeros).

- **Sentence 12** (Uncertainty Management): *"I should check if there's any leading zero affecting the bit count."* This sounds important — but resampling shows it actually *lowers* downstream accuracy. The check alone doesn't change the plan.
- **Sentence 13** (Plan Generation): *"Let me calculate the decimal value of 66666₁₆ and convert from there."* This is the thought anchor. When sentence 13 appears, the model almost always reaches the correct answer (19 bits). When it's replaced with something else, accuracy collapses.

The computation in sentences 14–40 (multiplying, converting, checking) just faithfully executes the plan set in sentence 13.

### Three-way convergence

Three independent methods all identify the same sentences as important:

| Method | What it measures | Finding |
|---|---|---|
| **Black-box Resampling** | Counterfactual impact on final answer | PG and UM sentences rank highest |
| **Receiver Head Analysis** | Which sentences specialized attention heads focus on | The same PG/UM sentences attract narrowly focused attention from all downstream tokens |
| **Causal Masking** | Direct effect of masking a sentence on downstream token logits | PG/UM sentences create the strongest causal links to future sentences |

The convergence across these three independent methods provides strong evidence that thought anchors are a real structural feature of LLM reasoning, not a measurement artifact.

### Why it matters for interpretability

Traditional interpretability looks at single forward passes (token-by-token activations). Thought anchor analysis operates at the **sentence level**, finding the coarse-grained decision structure of a reasoning trace — which steps set the agenda, which execute it, and which revise it. This level of abstraction is more useful for debugging reasoning failures: if a model gets a problem wrong, the error is usually traceable to a bad planning sentence, not to a downstream computation.

---

## Overview

Large reasoning models produce long chain-of-thought (CoT) traces before answering. Not every step matters equally — thought anchors (planning and backtracking sentences) have an outsized causal influence over the entire subsequent reasoning trajectory.

This project takes the **Thought Anchors** framework (Bogdan et al., 2025) and applies it to a new domain: **first-order logic reasoning** (FOLIO dataset), creating the first cross-domain comparison of thought anchors between math and logic. We also investigate whether the existing 8-tag function taxonomy transfers to logic CoT without modification.

### Research Questions

1. Do thought anchors exist in logic reasoning, and do they share structural properties with those in math?
2. Which functional categories (plan generation, backtracking, etc.) are most anchor-prone in logic vs. math?
3. Does the original 8-tag taxonomy transfer to logic CoT, and what is the minimal adaptation needed?



## What We Add Beyond the Original Repo

| Contribution | Description |
|---|---|
| **Data Generation** | Select 20 challenging problems from Logic domains; generat ~3,000 rollouts per problem with actual counts varying by reasoning length |
| **Cross-domain extension** | Apply all three Thought Anchors methods to FOLIO (logic) under settings aligned with the math experiments |
| **`taxonomy_transfer_analysis.py`** | New Exploratory Data Analysis(EDA) module: closed-set coverage rates, OOV diagnostics, and logic-proxy checks for the 8-tag taxonomy |
| **Pilot data & analysis** | 15,220 rollouts across Math + Logic; taxonomy coverage results reported |
| **End-to-end pipeline** | Data preparation → rollout generation → attribution analysis → report-ready artifacts |



## Methods

We compare three attribution methods from the original paper on both domains:

| Method | Type | How it works |
|---|---|---|
| **Black-box Resampling** | Black-box | Mask a CoT sentence, resample completions, measure accuracy / KL-divergence change |
| **Receiver Head Analysis** | White-box | Identify attention heads that act as "receivers" — aggregating cross-sentence information |
| **Causal Masking** | White-box | Directly suppress attention at specific layers, compare logit distributions |

The primary pipeline for this project is the **black-box** method, which requires only API access.


## Repository Structure

```
Beyond-Thought-Anchors/
│
├── generate_rollouts.py            # Step 1: Generate CoT rollouts with chunk masking
├── analyze_rollouts.py             # Step 2: Label chunks + compute importance metrics
├── step_attribution.py             # Step 3: Sentence-to-sentence causal attribution matrix
├── plots.py                        # Step 4: Generate paper-style figures
├── sentence_scatter_and_ttests.py  # Statistical tests (paired t-tests + scatter plots)
├── taxonomy_transfer_analysis.py   # NEW: taxonomy transfer diagnostics for logic domain
│
├── utils.py                        # Answer extraction, chunking, dataset loading
├── prompts.py                      # LLM auto-labeling prompts (8-tag classification)
├── selected_problems.json          # Curated MATH problems (25–75% accuracy range)
├── taxonomy_transfer_report.json   # Pilot output from taxonomy transfer diagnostics
├── readme_thought_anchors.md       # Detailed notes on original Thought Anchors codebase
│
├── masking_graphs/                 # MMLU masking / causal graph experiments (original paper infra)
│   └── resample/                   # Core resampling infrastructure (SuppObj, KL, API wrappers)
│
├── whitebox-analyses/              # White-box attention analyses (requires local GPU)
│   ├── pytorch_models/             # Model loading, hooks, ablation
│   ├── attention_analysis/         # Receiver-head identification and attention matrix extraction
│   └── scripts/                    # Experiment execution scripts
│
└── misc-experiments/               # Auxiliary scripts and notebooks
```



## Pilot Results

Pilot scale: **15,220 rollouts** (Math + Logic combined).

### Taxonomy Transfer (Logic Domain)

Results from `taxonomy_transfer_report.json`:

| Metric | Value |
|---|---|
| Base-tag coverage rate | **99.62%** |
| Unknown chunk rate | 0.38% |
| OOV chunk rate | **0.00%** |

**Conclusion:** The 8 top-level tags cover logic CoT almost completely. No out-of-vocabulary tags observed. Current recommendation: keep the 8-tag set; add logic-specific sub-tag guidelines only as needed.

### 8 Function Tags

Each CoT sentence is auto-labeled by GPT-4 into one of:

| Tag | Full Name |
|---|---|
| `AC` | Active Computation |
| `FR` | Fact Retrieval |
| `PG` | Plan Generation |
| `UM` | Uncertainty Management (backtracking) |
| `RC` | Result Consolidation |
| `SC` | Self Checking |
| `PS` | Problem Setup |
| `FAE` | Final Answer Emission |



## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file:

```bash
# Rollout generation — use any one provider
NOVITA_API_KEY=...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
OPENROUTER_API_KEY=...

# Required by analyze_rollouts.py for GPT-4 chunk labeling
OPENAI_API_KEY=...
```

> Local inference is also supported via `--provider Local` (requires GPU). DeepSeek-R1-Distill-Qwen-14B needs ~28 GB VRAM full precision or ~8 GB with 4-bit quantization (`-q`).



## Quick Start (Black-box Pipeline)

### Step 1 — Generate rollouts

```bash
python generate_rollouts.py \
  -m deepseek/deepseek-r1-distill-qwen-14b \
  -b correct \
  -r default \
  -o math_rollouts \
  -np 5 -nr 20 \
  -t 0.6 -tp 0.95 \
  -p Novita
```

Start small (`-np 5 -nr 20`) to verify the pipeline before scaling up.

**Output layout:**

```
math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/
├── correct_base_solution/
│   └── problem_N/
│       ├── problem.json
│       ├── base_solution.json
│       ├── chunks.json
│       ├── chunk_0/solutions.json   # rollouts from chunk 0
│       └── ...
├── incorrect_base_solution/
└── correct_base_solution_forced_answer/
```

### Step 2 — Analyze rollouts

```bash
python analyze_rollouts.py \
  -ic "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution" \
  -ii "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution" \
  -o "analysis/basic/deepseek-r1-distill-qwen-14b/alpha_1.0" \
  -im counterfactual_importance_kl
```

Outputs `analysis_results.json`, `chunks_labeled.json`, and summary plots.

### Step 3 — Step attribution

```bash
python step_attribution.py \
  -ad "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95" \
  -od "analysis/step_attribution" \
  -st 0.8 \
  -co
```

Produces an n×n causal importance matrix and in/out-degree charts — answering "who influences whom?"

### Step 4 — Plots

```bash
python plots.py -m qwen-14b --normalize
```

## Taxonomy Transfer Analysis (Our EDA Module)

`taxonomy_transfer_analysis.py` reports:

- Base-tag coverage rate (8-tag closed set)
- Unknown / OOV rates
- Per-tag frequency and mean CF-KL importance score
- Logic-proxy diagnostics and adaptation recommendations

```bash
python taxonomy_transfer_analysis.py \
  --analysis_root "<path_to_analysis_output>" \
  --output_json "taxonomy_transfer_report.json"
```



## Planned Final-Scale Experiments

- **20–30 FOLIO problems** (logic) × correct / incorrect CoTs
- **~100 rollouts per chunk**
- Compare all three attribution methods (black-box, receiver-head, causal masking) on Math vs. Logic
- Analyze anchor distributions by functional tag across domains
- Analyze cross methods consistency


## References

- Bogdan et al. (2025). *Thought Anchors: Which LLM Reasoning Steps Matter?* https://arxiv.org/abs/2506.19143
- Han et al. (2022). *FOLIO: Natural Language Reasoning with First-Order Logic* https://arxiv.org/abs/2209.00840



## Course Context

- Course: Deep Learning for Natural Language Processing (CS639), UW-Madison
- This repository supports both the proposal-stage pilot and final-scale experiments.
