#!/bin/bash

cd /Users/sylvia/anaconda_projects/PythonProject/thought-anchors
source .venv/bin/activate

echo "========== Step 1: Prepare Caches =========="
python whitebox-analyses/scripts/prep_attn_cache.py \
  --model qwen-14b \
  --max-problems 5 \
  --verbose \
  --proximity-ignore 4 \
  --control-depth \
  --top-k 20 \
  --skip-attention

echo "========== Step 2: Evaluate Receiver Head Reliability =========="
python whitebox-analyses/scripts/eval_receiver_head_reliability.py \
  --model-name qwen-14b \
  --proximity-ignore 4 \
  --control-depth \
  --output-dir plots/receiver_head_reliability \
  --dpi 300

echo "========== Step 3: Plot Attention Heads (Problem 1591) =========="
python whitebox-analyses/scripts/plot_attention_heads.py \
  --problem-num 1591 \
  --correct \
  --model-name qwen-14b \
  --layer 20 \
  --proximity-ignore 4 \
  --control-depth \
  --output-dir plots/head_distributions/problem_1591 \
  --dpi 300

echo "========== Step 4: Plot Top-K Receiver Heads =========="
python whitebox-analyses/scripts/plot_attention_heads.py \
  --problem-num 1591 \
  --correct \
  --top-k 10 \
  --model-name qwen-14b \
  --proximity-ignore 4 \
  --control-depth \
  --output-dir plots/head_distributions/problem_1591_topk \
  --dpi 300

echo "========== Step 5: Plot Kurtosis Statistics =========="
python whitebox-analyses/scripts/plot_kurt_stats.py \
  --model-name qwen-14b \
  --proximity-ignore 4 \
  --control-depth \
  --output-dir plots/kurtosis_analysis \
  --dpi 300

echo "========== Step 6: Generate Receiver Head CSVs =========="
python whitebox-analyses/scripts/generate_rec_csvs.py \
  --model-name qwen-14b \
  --proximity-ignore 4 \
  --control-depth \
  --top-k 20 \
  --output-dir results/receiver_heads

echo "========== Step 7: Prepare Suppression Matrices =========="
python whitebox-analyses/scripts/prep_suppression_mtxs.py \
  --model qwen-14b \
  --max-problems 5 \
  --skip-vertical \
  --skip-receiver \
  --verbose

echo "========== Step 8: Plot Suppression Matrix =========="
python whitebox-analyses/scripts/plot_suppression_matrix.py \
  --model-name qwen-14b \
  --problem-num 1591 \
  --is-correct \
  --output-dir plots/suppression_matrix/problem_1591 \
  --dpi 300

echo "========== Step 9: Plot Suppression Line =========="
python whitebox-analyses/scripts/plot_suppression_line_unused.py \
  --model-name qwen-14b \
  --problem-num 1591 \
  --is-correct \
  --output-dir plots/suppression_line/problem_1591 \
  --dpi 300

echo "========== Step 10: Evaluate Same Sentences =========="
python whitebox-analyses/scripts/eval_receiver_head_same_sentences.py \
  --model-name qwen-14b \
  --proximity-ignore 4 \
  --control-depth \
  --output-dir plots/receiver_head_consistency \
  --dpi 300

echo "========== Step 11: Plot Receiver Head Taxonomy =========="
python whitebox-analyses/scripts/plot_rec_taxonomy.py \
  --model-name qwen-14b \
  --top-k 30 \
  --output-dir plots/receiver_head_taxonomy \
  --dpi 300

echo "========== Step 12: Print Case Study =========="
python whitebox-analyses/scripts/print_case_study_transcript.py \
  --model-name qwen-14b \
  --problem-num 1591 \
  --is-correct \
  --top-k 5

echo "========== All Steps Complete! =========="
