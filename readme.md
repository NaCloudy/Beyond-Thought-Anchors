# Beyond Thought Anchors

Beyond Thought Anchors: Cross-Domain and Cross-Method Analysis of LLM Reasoning Steps

> **CS639 Deep Learning for NLP — Course Project**
>
> Extending _Thought Anchors: Which LLM Reasoning Steps Matter?_ from math reasoning to logic reasoning.

## What is a Thought Anchor?

When a reasoning model solves a problem, it generates a long chain-of-thought (CoT) trace — hundreds of sentences of internal deliberation. The central question of the original paper is: **which of those sentences actually matter?**

### The counterintuitive finding

The naive answer is "computation sentences" (Active Computation — where the model does actual algebra or arithmetic). If you measure importance by _interrupting_ the model mid-trace and forcing an immediate answer, computation steps look most important, because they're closest to the answer.

But this confounds two things: _which sentence contains the answer_ vs. _which sentence caused the model to find the correct answer_.

The paper's key insight is that once a reasoning model has committed to a particular plan, the subsequent computations are **largely predetermined** — the model will reliably execute them. The real fork point is earlier: _which plan did the model adopt?_

### What actually matters: Planning and Backtracking

When importance is measured correctly — by **counterfactually resampling the entire subsequent chain** (regenerating all downstream steps, not just forcing the next token) — a different picture emerges:

- **Plan Generation sentences** (e.g., _"Let me calculate the decimal value first and convert from there"_) have the highest counterfactual importance. They set the trajectory; once this sentence appears, the model reliably executes the chosen plan and reaches a specific answer.
- **Uncertainty Management / Backtracking sentences** (e.g., _"Wait, I should double-check this"_) are the second most important. They are pivot points where the model abandons a failing approach and switches direction.
- **Active Computation sentences** drop sharply in importance under this measure. They are executing a plan, not choosing one — if you remove them, the model generates equivalent computations anyway.

These high-impact planning and backtracking sentences are the **thought anchors**: they anchor the trajectory of the entire subsequent reasoning.

### The case study that illustrates it

Problem: _"How many bits does 66666₁₆ have in base 2?"_

The model initially plans to multiply 5 hex digits × 4 bits/digit = 20 bits. This plan is wrong (it ignores leading zeros).

- **Sentence 12** (Uncertainty Management): _"I should check if there's any leading zero affecting the bit count."_ This sounds important — but resampling shows it actually _lowers_ downstream accuracy. The check alone doesn't change the plan.
- **Sentence 13** (Plan Generation): _"Let me calculate the decimal value of 66666₁₆ and convert from there."_ This is the thought anchor. When sentence 13 appears, the model almost always reaches the correct answer (19 bits). When it's replaced with something else, accuracy collapses.

The computation in sentences 14–40 (multiplying, converting, checking) just faithfully executes the plan set in sentence 13.

### Three-way convergence

Three independent methods all identify the same sentences as important:

| Method                     | What it measures                                               | Finding                                                                                |
| -------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Black-box Resampling**   | Counterfactual impact on final answer                          | PG and UM sentences rank highest                                                       |
| **Receiver Head Analysis** | Which sentences specialized attention heads focus on           | The same PG/UM sentences attract narrowly focused attention from all downstream tokens |
| **Causal Masking**         | Direct effect of masking a sentence on downstream token logits | PG/UM sentences create the strongest causal links to future sentences                  |

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

| Contribution                        | Description                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Generation**                 | Select 20 challenging problems from Logic domains; generat ~3,000 rollouts per problem with actual counts varying by reasoning length |
| **Cross-domain extension**          | Apply all three Thought Anchors methods to FOLIO (logic) under settings aligned with the math experiments                             |
| **`taxonomy_transfer_analysis.py`** | New Exploratory Data Analysis(EDA) module: closed-set coverage rates, OOV diagnostics, and logic-proxy checks for the 8-tag taxonomy  |
| **Pilot data & analysis**           | 15,220 rollouts across Math + Logic; taxonomy coverage results reported                                                               |
| **End-to-end pipeline**             | Data preparation → rollout generation → attribution analysis → report-ready artifacts                                                 |

## Methods

We compare three attribution methods from the original paper on both domains:

| Method                     | Type      | How it works                                                                              |
| -------------------------- | --------- | ----------------------------------------------------------------------------------------- |
| **Black-box Resampling**   | Black-box | Mask a CoT sentence, resample completions, measure accuracy / KL-divergence change        |
| **Receiver Head Analysis** | White-box | Identify attention heads that act as "receivers" — aggregating cross-sentence information |
| **Causal Masking**         | White-box | Directly suppress attention at specific layers, compare logit distributions               |

The primary pipeline for this project is the **black-box** method, which requires only API access.

## Repository Structure

```
Beyond-Thought-Anchors/
│
├── readme.md                       # This file
├── LICENSE.md
├── requirements.txt
├── reproduce.ipynb                 # White-box pipeline entry (Method 2 + 3)
├── gpqa_pilot.ipynb                # Pilot data exploration
├── gpqa_20_candidates.json         # 20 selected GPQA problems
│
│   # === Method 1 — Black-box Resampling ===
├── generate_rollouts.py            # Step 1: Generate CoT rollouts with chunk masking
├── analyze_rollouts.py             # Step 2: Label chunks + compute importance metrics
├── step_attribution.py             # Step 3: Sentence-to-sentence causal attribution matrix
├── plots.py                        # Step 4: Generate paper-style figures
├── sentence_scatter_and_ttests.py  # Statistical tests (paired t-tests + scatter plots)
├── utils.py                        # Answer extraction, chunking, dataset loading
├── prompts.py                      # LLM auto-labeling prompts (8-tag classification)
│
│   # === Cross-domain & cross-method analysis (Exp 2/3) ===
├── cross_domain_analysis_new.py    # Latest Exp 2/3 analysis script
├── cross_domain_analysis.py        # Archived previous version of Exp 2/3 analysis
│
├── eda/                            # Exploratory data analysis
│   ├── taxonomy_transfer_analysis.py  # NEW: 8-tag closed-set coverage / OOV diagnostics
│   └── plot_pilot.ipynb
│
├── results/                        # Exp 2/3 statistical results
│   ├── README.md                   # Detailed JSON schema
│   └── simple_results.md           # Summary table for report reference
│
├── masking_graphs/                 # MMLU masking / causal graph experiments (original paper infra)
│   └── resample/                   # Core resampling infrastructure (SuppObj, KL, API wrappers)
│
├── whitebox-analyses/              # White-box attention analyses
│   ├── README_on_pkld_decorator.md
│   ├── pytorch_models/             # Model loading, hooks, ablation
│   ├── attention_analysis/         # Receiver-head identification and attention matrix extraction
│   └── scripts/                    # Experiment execution scripts (used by reproduce.ipynb)
│
├── misc-experiments/               # Auxiliary scripts and notebooks
└── misc-scripts/                   # Hugging Face dataset upload/download utilities
```

## Pilot Results

Pilot scale: **15,220 rollouts** (Math + Logic combined).

### Taxonomy Transfer (Logic Domain)

Results from running `eda/taxonomy_transfer_analysis.py`:

| Metric                 | Value      |
| ---------------------- | ---------- |
| Base-tag coverage rate | **99.62%** |
| Unknown chunk rate     | 0.38%      |
| OOV chunk rate         | **0.00%**  |

**Conclusion:** The 8 top-level tags cover logic CoT almost completely. No out-of-vocabulary tags observed. Current recommendation: keep the 8-tag set; add logic-specific sub-tag guidelines only as needed.

### 8 Function Tags

Each CoT sentence is auto-labeled by GPT-4 into one of:

| Tag   | Full Name                             |
| ----- | ------------------------------------- |
| `AC`  | Active Computation                    |
| `FR`  | Fact Retrieval                        |
| `PG`  | Plan Generation                       |
| `UM`  | Uncertainty Management (backtracking) |
| `RC`  | Result Consolidation                  |
| `SC`  | Self Checking                         |
| `PS`  | Problem Setup                         |
| `FAE` | Final Answer Emission                 |

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

## Quick Start

### Step 0 — Select pilot problems (one-time, Google Colab A100 GPU)

`gpqa_pilot.ipynb` runs the base reasoning model on the GPQA Diamond split, scores each question by base accuracy, and selects 20 challenging problems (those in the harder accuracy band) as the pilot set. The selected problem IDs are written to `gpqa_20_candidates.json`, which all downstream scripts consume.

```text
Environment   : Google Colab, A100 GPU
Notebook      : gpqa_pilot.ipynb
Output (committed) : gpqa_20_candidates.json
```

> Already committed — you only need to re-run this if you want to refresh the pilot set.

---

### Method 1 — Black-box Resampling

#### Step 1 — Generate rollouts

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

#### Step 2 — Analyze rollouts

```bash
python analyze_rollouts.py \
  -ic "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution" \
  -ii "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution" \
  -o "analysis/basic/deepseek-r1-distill-qwen-14b/alpha_1.0" \
  -im counterfactual_importance_kl
```

Outputs `analysis_results.json`, `chunks_labeled.json`, and summary plots.

#### Step 3 — Step attribution

```bash
python step_attribution.py \
  -ad "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95" \
  -od "analysis/step_attribution" \
  -st 0.8 \
  -co
```

Produces an n×n causal importance matrix and in/out-degree charts — answering "who influences whom?"

#### Step 4 — Plots

```bash
python plots.py -m qwen-14b --normalize
```

---

### Method 2 — Receiver Head Analysis (requires local GPU)

> **Notebook entry point:** `reproduce.ipynb` orchestrates Method 2 + Method 3 end-to-end on a single GPU machine. The commands below mirror the notebook for users who prefer the CLI.

`prep_attn_cache.py` loads the 14B model once per problem, runs a single forward pass, and caches the full 48 × 40 attention tensors to `attn_cache/`. `generate_rec_csvs.py` then reads those cached tensors and computes receiver-head scores — no model loading needed at this stage.

```bash
# Step 1 — Cache attention matrices (1 forward pass per problem; needs GPU)
python whitebox-analyses/scripts/prep_attn_cache.py --model qwen-14b --dataset gpqa --skip-receiver
python whitebox-analyses/scripts/prep_attn_cache.py --model qwen-14b --dataset math --skip-receiver

# Step 2 — Compute receiver head scores → CSV (CPU-only)
python whitebox-analyses/scripts/generate_rec_csvs.py \
    --model-name qwen-14b --data-model qwen-14b --dataset gpqa \
    --top-k 32 --proximity-ignore 16 --output-dir csvs/gpqa
python whitebox-analyses/scripts/generate_rec_csvs.py \
    --model-name qwen-14b --data-model qwen-14b --dataset math \
    --top-k 32 --proximity-ignore 16 --output-dir csvs/math
```

Outputs:

- `csvs/{gpqa,math}/receiver_head_scores_correct_qwen-14b_k32_pi16.csv` — one row per sentence with vertical receiver-head scores.

---

### Method 3 — Causal Masking (requires local GPU)

For every sentence in every problem, mask that sentence's attention and re-run the forward pass; the token-level KL divergence against the unmasked baseline yields a sentence × sentence causal influence matrix. The row sum gives per-sentence causal importance.

```bash
# N_sentences forward passes per problem (~100 sentences/problem on GPQA)
python whitebox-analyses/scripts/prep_suppression_mtxs.py \
    --model-name qwen-14b --dataset gpqa --output-dir kl_results/gpqa
python whitebox-analyses/scripts/prep_suppression_mtxs.py \
    --model-name qwen-14b --dataset math --output-dir kl_results/math
```

Outputs:

- `kl_results/{gpqa,math}/qwen-14b/correct/problem_N_kl.npy` — sentence × sentence KL matrix per problem.

> `@pkld` caching: interrupted runs resume automatically; problems already finished are skipped. See `whitebox-analyses/README_on_pkld_decorator.md` for details.

---

### Plotting (CPU-only, all post-hoc)

After Methods 2 and 3 finish, all plotting scripts read from `attn_cache/`, `csvs/`, and `kl_results/` — no model loading and no GPU needed. Refer to `reproduce.ipynb` (Plotting section) for the full set of figure-generation commands (kurtosis stats, taxonomy boxplots, attention-head curves, suppression heatmaps, suppression top-k lines, split-half reliability).

---

## Taxonomy Transfer Analysis (Our EDA Module)

`taxonomy_transfer_analysis.py` reports:

- Base-tag coverage rate (8-tag closed set)
- Unknown / OOV rates
- Per-tag frequency and mean CF-KL importance score
- Logic-proxy diagnostics and adaptation recommendations

```bash
python eda/taxonomy_transfer_analysis.py \
  --analysis_root "<path_to_analysis_output>" \
  --output_json "taxonomy_transfer_report.json"
```
