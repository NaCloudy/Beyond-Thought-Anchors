# Beyond Thought Anchors

Code for reproducing and analyzing thought anchors in large language model reasoning across MATH and GPQA Diamond.

This repository extends the _Thought Anchors_ framework from math reasoning to science question answering. It reproduces the three original analysis methods on MATH, runs the same three methods on GPQA Diamond, and provides a shared statistical pipeline for comparing domains and methods.

The completed project uses **GPQA Diamond**, not FOLIO. The final experiments and analysis are MATH vs. GPQA.

## Overview

Reasoning models often generate long chains of thought before producing an answer. Thought-anchor analysis asks which intermediate reasoning sentences have the largest downstream influence.

This codebase covers three attribution methods:

| Method   | Name                   | Signal                                                                               |
| -------- | ---------------------- | ------------------------------------------------------------------------------------ |
| Method 1 | Black-box resampling   | Counterfactual change in rollout accuracy or KL after replacing a reasoning step     |
| Method 2 | Receiver-head analysis | Attention heads whose vertical attention concentrates on influential prior sentences |
| Method 3 | Causal masking         | KL change caused by suppressing attention to a sentence during later computation     |

The final analysis compares:

- domain structure: MATH vs. GPQA Diamond
- method agreement: black-box resampling vs. receiver-head scores vs. causal masking
- sentence-function categories: planning, uncertainty management, computation, fact retrieval, and related tags

## Repository Structure

```text
Beyond-Thought-Anchors/
|-- readme.md                         # Project entry point
|-- LICENSE.md
|-- requirements.txt
|
|-- generate_rollouts.py              # Method 1: MATH base solutions and counterfactual rollouts
|-- analyze_rollouts.py               # Method 1: label chunks and compute importance metrics
|-- step_attribution.py               # Method 1: sentence-to-sentence attribution matrix
|-- plots.py                          # Method 1 plotting utilities
|-- sentence_scatter_and_ttests.py    # Additional Method 1 statistics and scatter plots
|
|-- reproduce.ipynb                   # Reproduce Methods 2 and 3 on MATH and GPQA
|-- cross_domain_analysis.py          # Final Exp 2/3 cross-domain and cross-method analysis
|
|-- gpqa_pilot.ipynb                  # GPQA screening notebook
|-- gpqa_20_candidates.json           # Selected GPQA Diamond subset
|-- math_selected_problems.json       # Selected MATH subset
|
|-- prompts.py                        # Prompts for sentence-function labeling
|-- utils.py                          # Answer extraction, chunking, dataset helpers
|
|-- eda/
|   |-- taxonomy_transfer_analysis.py # Exploratory tag-coverage diagnostics
|   `-- plot_pilot.ipynb              # plots for proposal reports
|
|-- whitebox-analyses/
|   |-- README_on_pkld_decorator.md   # Notes on pkld caching and resumable runs
|   |-- attention_analysis/           # Receiver-head utilities
|   |-- pytorch_models/               # Model loading, hooks, and ablations
|   `-- scripts/
|       |-- prep_attn_cache.py        # Method 2: cache attention tensors
|       |-- generate_rec_csvs.py      # Method 2: receiver-head score CSVs
|       |-- prep_suppression_mtxs.py  # Method 3: causal masking KL matrices
|       |-- plot_kurt_stats.py
|       |-- plot_rec_taxonomy.py
|       |-- plot_attention_heads.py
|       |-- plot_suppression_matrix.py
|       |-- plot_suppression_line_unused.py
|       `-- eval_receiver_head_same_sentences.py
|
|-- masking_graphs/                   # Resampling and masking infrastructure from the original codebase
|-- misc-experiments/                 # Auxiliary experiments
|-- misc-scripts/                     # Dataset upload/download helpers
|
|-- math-rollouts/                    # Generated Method 1 MATH rollouts
|-- gpqa-rollouts/                    # Generated Method 1 GPQA rollouts
|-- analysis/                         # Method 1 analyzed outputs
|-- attn_cache/                       # Method 2 cached attention tensors
|-- csvs/                             # Method 2 receiver-head CSV outputs
|-- kl_results/                       # Method 3 KL matrices
|-- plots/                            # Generated figures
`-- results/                          # Final cross-domain analysis outputs and documentation
    `-- README.md                     # Schema and metric notes for Exp 2/3 results
```

Generated artifact directories are intentionally treated as local experiment outputs. They can be regenerated from the scripts and notebooks below.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For API-based rollout generation and chunk labeling, create a `.env` file:

```bash
OPENAI_API_KEY=...
NOVITA_API_KEY=...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
OPENROUTER_API_KEY=...
```

`generate_rollouts.py` supports API providers and local inference. Methods 2 and 3 require local GPU execution with the Qwen-14B distilled reasoning model. The notebook uses a separate Hugging Face cache path on larger disks when running on cloud machines.

## Data

The final experiments use two 20-problem subsets:

- MATH: selected problems stored in `math_selected_problems.json`
- GPQA Diamond: selected candidate questions stored in `gpqa_20_candidates.json`

`gpqa_pilot.ipynb` documents the GPQA screening workflow. If you only want to reproduce the final experiments, use the committed selection files directly.

## Method 1: Black-Box Resampling

Method 1 generates a base chain of thought, chunks it into sentence-level steps, resamples downstream reasoning after each step, and measures how much final-answer behavior changes.

The checked-in rollout generator is MATH-oriented: it loads problems through `load_math_problems` in `utils.py`. Generate a small MATH rollout run with:

```bash
python generate_rollouts.py \
  -m deepseek/deepseek-r1-distill-qwen-14b \
  -b correct \
  -r default \
  -o math-rollouts \
  -np 5 \
  -nr 20 \
  -t 0.6 \
  -tp 0.95 \
  -p Novita
```

Generate incorrect-base and forced-answer variants as needed by changing `-b` and `-r`.

For the final cross-domain analysis, GPQA Method 1 artifacts must follow the same analyzed-output schema under `analysis/gpqa/correct_base_solution/`. The final comparison script consumes the analyzed JSON and does not require rollout generation to be rerun.

Analyze rollout outputs:

```bash
python analyze_rollouts.py \
  -ic "math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution" \
  -ii "math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution" \
  -o "analysis/math/correct_base_solution" \
  -im counterfactual_importance_kl
```

Common follow-up commands:

```bash
python step_attribution.py \
  -ad "math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95" \
  -od "analysis/step_attribution" \
  -st 0.8 \
  -co

python plots.py -m qwen-14b --normalize
```

Primary Method 1 output expected by the final analysis:

```text
analysis/{domain}/correct_base_solution/analysis_results.json
```

where `{domain}` is `math` or `gpqa`.

## Methods 2 and 3: White-Box Reproduction

`reproduce.ipynb` is the main entry point for Methods 2 and 3. It reproduces both methods on GPQA and MATH, verifies expected output files, and runs CPU-only plotting scripts after the GPU-heavy computations are complete.

### Method 2: Receiver-Head Analysis

Cache attention matrices:

```bash
python whitebox-analyses/scripts/prep_attn_cache.py \
  --model qwen-14b \
  --dataset gpqa \
  --skip-receiver

python whitebox-analyses/scripts/prep_attn_cache.py \
  --model qwen-14b \
  --dataset math \
  --skip-receiver
```

Generate receiver-head score CSVs:

```bash
python whitebox-analyses/scripts/generate_rec_csvs.py \
  --model-name qwen-14b \
  --data-model qwen-14b \
  --dataset gpqa \
  --top-k 32 \
  --proximity-ignore 16 \
  --output-dir csvs/gpqa

python whitebox-analyses/scripts/generate_rec_csvs.py \
  --model-name qwen-14b \
  --data-model qwen-14b \
  --dataset math \
  --top-k 32 \
  --proximity-ignore 16 \
  --output-dir csvs/math
```

Primary Method 2 outputs:

```text
csvs/gpqa/receiver_head_scores_correct_qwen-14b_k32_pi16.csv
csvs/math/receiver_head_scores_correct_qwen-14b_k32_pi16.csv
```

### Method 3: Causal Masking

Compute sentence-by-sentence KL matrices:

```bash
python whitebox-analyses/scripts/prep_suppression_mtxs.py \
  --model-name qwen-14b \
  --dataset gpqa \
  --output-dir kl_results/gpqa

python whitebox-analyses/scripts/prep_suppression_mtxs.py \
  --model-name qwen-14b \
  --dataset math \
  --output-dir kl_results/math
```

Primary Method 3 outputs:

```text
kl_results/gpqa/qwen-14b/correct/problem_N_kl.npy
kl_results/math/qwen-14b/correct/problem_N_kl.npy
```

The `pkld` cache used by the white-box scripts makes interrupted runs resumable. See `whitebox-analyses/README_on_pkld_decorator.md` for details.

### White-Box Plotting

After `attn_cache/`, `csvs/`, and `kl_results/` exist, plotting is CPU-only. The notebook includes commands for:

- receiver-head kurtosis plots: `plot_kurt_stats.py`
- receiver score by sentence taxonomy: `plot_rec_taxonomy.py`
- per-problem receiver-head curves: `plot_attention_heads.py`
- causal masking heatmaps: `plot_suppression_matrix.py`
- top-k suppression lines: `plot_suppression_line_unused.py`
- split-half reliability: `eval_receiver_head_same_sentences.py`

## Final Cross-Domain Analysis

Run the final statistical comparison from the repository root:

```bash
python cross_domain_analysis.py
```

Inputs expected by the script:

```text
analysis/{domain}/correct_base_solution/analysis_results.json
csvs/{domain}/receiver_head_scores_correct_qwen-14b_k32_pi16.csv
kl_results/{domain}/qwen-14b/correct/problem_N_kl.npy
```

Output:

```text
results/cross_domain_results.json
```

The script also prints a console summary. Detailed result schema and metric definitions live in `results/README.md`.

At a high level, the output contains four experiment blocks:

| Key     | Experiment        | Contents                            |
| ------- | ----------------- | ----------------------------------- |
| `exp2a` | Category profile  | Gini coefficient and JSD            |
| `exp2b` | Attention entropy | Receiver-head score entropy         |
| `exp2c` | Causal topology   | Causal reach ratio and anchor shift |
| `exp3`  | Method agreement  | Kendall's W, tau, and top-k overlap |

Method names in `exp3`:

```text
m1 = Method 1, black-box resampling with CF-KL
m2 = Method 2, receiver-head score
m3 = Method 3, causal masking with fixed k=3 forward-window mean
```

### Current Summary Statistics

The current final-analysis results can be summarized as:

- Category concentration is similar across domains: Gini difference is not significant (`p = 0.48`, Cliff's delta `0.11`).
- Category distribution differs moderately: Jensen-Shannon divergence is `0.095`, driven mainly by plan generation, self-checking, and uncertainty-management frequency shifts.
- Receiver-head attention entropy is slightly higher on GPQA (`6.24` vs. `6.05`) but not significant (`p = 0.35`).
- Causal reach ratio does not differ significantly (`p = 0.47`).
- Anchor location is the clearest cross-domain difference: GPQA anchors occur later in the reasoning trace on average (`0.36` vs. `0.28`, `p = 0.0009`, small effect).
- Three-way method agreement is moderate in both domains: Kendall's W is `0.332` on GPQA and `0.346` on MATH.
- Method 1 and Method 2 are the most consistent pair; Method 2 and Method 3 are largely uncorrelated.

## Sentence Function Tags

The analysis uses the original eight top-level function tags:

| Tag   | Function               |
| ----- | ---------------------- |
| `AC`  | Active computation     |
| `FR`  | Fact retrieval         |
| `PG`  | Plan generation        |
| `UM`  | Uncertainty management |
| `RC`  | Result consolidation   |
| `SC`  | Self-checking          |
| `PS`  | Problem setup          |
| `FAE` | Final answer emission  |

`eda/taxonomy_transfer_analysis.py` is retained as an exploratory diagnostic for tag coverage and out-of-vocabulary labeling behavior. It is not the main final-analysis entry point.

Example:

```bash
python eda/taxonomy_transfer_analysis.py \
  --analysis_root "analysis" \
  --output_json "eda/taxonomy_transfer_report.json"
```

## Notes for Reproduction

- Start with small `-np` and `-nr` values when testing rollout generation; full runs are API- and GPU-expensive.
- Methods 2 and 3 are much heavier than Method 1. Method 2 caches one full attention pass per problem; Method 3 reruns masked forward passes for every sentence.
- `reproduce.ipynb` is the most reliable guide for Methods 2 and 3 because it includes expected-file checks after each stage.
- `cross_domain_analysis.py` assumes the final output paths shown above. If you store artifacts elsewhere, either copy them into the expected layout or edit the constants in that script.

## License

This repository is released under the license in `LICENSE.md`.
