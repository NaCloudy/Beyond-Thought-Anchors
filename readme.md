# Beyond Thought Anchors

Extending **Thought Anchors: Which LLM Reasoning Steps Matter?** from math reasoning to logic reasoning.

## 1. Project Overview

This repository is our CS639 course project codebase, built on **Thought Anchors: Which LLM Reasoning Steps Matter?**.

Core project focus:

- Apply the original Thought Anchors codebase to the logic dataset **FOLIO** and build matched math/logic experimental settings.
- For final-scale experiments, target **20-30 FOLIO problems**; for each problem, generate correct / incorrect and forced-answer CoTs, then sample rollouts per chunk (target structure: ~100 rollouts/chunk).
- Run and compare the three methods from the paper on logic + math: **Black-box Resampling**, **Receiver Head Analysis**, and **Causal Masking**.
- Analyze thought-anchor characteristics in the logic domain and compare them against the math domain.

EDA sub-question (not the core objective):

> Can the math-oriented 8-tag taxonomy transfer to logic CoT, and if not, what is the minimal adaptation while preserving cross-domain comparability?

## 2. What We Add Beyond the Original Repo

- Cross-domain extension from math-only analysis to **Math + Logic** under aligned settings.
- End-to-end project pipeline for coursework delivery: data preparation, rollout generation, attribution analysis, and report-ready artifacts.
- **NEW: `taxonomy_transfer_analysis.py`** as a focused EDA module for closed-set coverage, unknown/OOV rates, and logic-proxy checks.

## 3. Repository Structure

```text
Beyond-Thought-Anchors/
├── generate_rollouts.py            # Generate CoT rollouts with chunk masking
├── analyze_rollouts.py             # Label chunks + compute importance metrics
├── step_attribution.py             # Step-to-step attribution matrix (i -> j)
├── taxonomy_transfer_analysis.py   # NEW: taxonomy transfer diagnostics
├── taxonomy_transfer_report.json   # Example output from transfer diagnostics
├── readme_thought_anchors.md       # Notes from original Thought Anchors repo/paper
├── selected_problems.json          # Selected MATH problems
├── masking_graphs/                 # MMLU masking/graph experiments
├── whitebox-analyses/              # Receiver-head and causal/attention analyses
└── misc-experiments/               # Additional scripts
```

The above are the main scripts and directories of this repository, among which `taxonomy_transfer_analysis.py` is a key new addition to this project.

## 4. Environment Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Create `.env` (required keys depend on your pipeline):

```bash
# For rollout generation via APIs (choose any provider)
NOVITA_API_KEY=...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
OPENROUTER_API_KEY=...

# Required by analyze_rollouts.py for chunk labeling
OPENAI_API_KEY=...
```

## 5. Quick Start (Black-box Pipeline)

### Step 1. Generate rollouts

```bash
python generate_rollouts.py \
  -m deepseek/deepseek-r1-distill-qwen-14b \
  -b correct \
  -r default \
  -o math_rollouts \
  -np 5 \
  -nr 20 \
  -t 0.6 \
  -tp 0.95 \
  -p Novita
```

First, conduct small-scale tests (e.g., `-np 5 -nr 20`) to confirm the process before scaling up to a formal experiment.

### Step 2. Analyze rollouts | 分析 rollouts

```bash
python analyze_rollouts.py \
  -ic "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution" \
  -ii "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution" \
  -o "analysis/basic/deepseek-r1-distill-qwen-14b/alpha_1.0" \
  -im counterfactual_importance_kl
```

This step will output `analysis_results.json`, `chunks_labeled.json`, and various visual statistical results.

### Step 3. Step attribution

```bash
python step_attribution.py \
  -ad "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95" \
  -od "analysis/step_attribution" \
  -st 0.8 \
  -co
```

Output the step-to-step importance matrix and in-degree/out-degree graph, answering the question "Who influences whom?".

## 6. NEW: Taxonomy Transfer Analysis

`taxonomy_transfer_analysis.py` is our project-specific extension.

It reports:

- base-tag coverage rate (8-tag closed set)
- unknown/OOV rates
- per-tag frequency and mean CF-KL
- logic proxy diagnostics and adaptation recommendation

Run:

```bash
python taxonomy_transfer_analysis.py \
  --analysis_root "D:\Files\UW-M\26_1_spring\CS639\prj\proposal_math+logic" \
  --output_json "taxonomy_transfer_report.json"
```

## 7. Pilot Snapshot

- Total pilot rollouts: **15,220** (Math + Logic).
- Logic aggregate coverage from `taxonomy_transfer_report.json`:
  - `base_tag_coverage_rate = 0.9962`
  - `unknown_chunk_rate = 0.0038`
  - `oov_chunk_rate = 0.0000`  
    The logic-side aggregation results show that the 8 tags are almost completely covered, and there are no out-of-vocabulary (OOV) values.
- Current recommendation: keep 8 top-level tags; add logic-specific guideline/sub-tags only when needed.

## 8. Course Project Context

- Course: Deep Learning for Natural Language Processing (CS639).
- This repo supports both proposal-stage pilot and final-stage scale-up experiments.

Local paths used in our current setup:

- Code: `D:\Repos\courses\Beyond-Thought-Anchors`
- Pilot data: `D:\Files\UW-M\26_1_spring\CS639\prj\proposal_pilot_results`
- Pilot analysis: `D:\Files\UW-M\26_1_spring\CS639\prj\proposal_math+logic`

## 9. References

- Bogdan et al. (2025). _Thought Anchors: Which LLM Reasoning Steps Matter?_  
  https://arxiv.org/abs/2506.19143
- Han et al. (2022). _FOLIO: Natural Language Reasoning with First-Order Logic_  
  https://arxiv.org/abs/2209.00840

## 10. License

This repository follows the license terms of the upstream project where applicable (see `LICENSE.md`).
