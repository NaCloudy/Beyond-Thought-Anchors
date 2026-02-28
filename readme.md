## Dataset

### `math-rollouts`

- source: https://huggingface.co/datasets/uzaymacar/math-rollouts

```
math_rollouts/
├── deepseek-r1-distill-llama-8b/
  └── temperature_0.6_top_p_0.95/
    ├── correct_base_solution/ # 模型通过推理得出正确答案的问题
        # 每个问题存储在名为 problem_X/ 的目录中，其中 X 是问题 ID：
        ├──problem_330/
            ├── problem.json # Problem statement and metadata
                {
                "problem": "Compute $3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3(1+3)))))))))$",
                "level": "Level 5",
                "type": "Algebra",
                "gt_solution": "Not to be tricked by the excess of parentheses...",
                "gt_answer": "88572",
                "nickname": "Nested Multiplication"
                }
            ├── chunks_labeled.json # Importance and analysis data for each reasoning step
                {
                "chunk": "First, I notice that the expression is a nested set of terms...",
                "chunk_idx": 6,
                "function_tags": ["fact_retrieval"],
                "depends_on": ["3"], # 对之前区块的依赖
                "accuracy": 0.7448979591836735, # 包含该块时的成功率
                "resampling_importance_accuracy": 0.14399092970521532, # 通过重采样测量的重要性
                "resampling_importance_kl": 1.986370430176875,
                "counterfactual_importance_accuracy": -0.09941520467836251, # 通过反事实衡量的重要性
                "counterfactual_importance_kl": 1.19399334472036,
                "forced_importance_accuracy": 0.0, # 强制特定完成时的重要性
                "forced_importance_kl": 1.666184462112768,
                "different_trajectories_fraction": 0.7021276595744681, # 此时推理路径的多样性
                "overdeterminedness": 0.5, # 当前步骤的过度确定程度（即重新采样的句子与原句子的相似度）
                "summary": "identify nested expression"
                }
            ├── chunk_0/
            │ └── solutions.json # Multiple solution attempts for chunk 0
                {
                "chunk_resampled": "Let me work from the innermost expression outward...", # 被重采样的chunk，可能是逗号分隔？
                "is_correct": true, # 回答正确还是错误
                "answer": "88572",
                "rollout" : "Solve this math problem step by step..." # 模型从think之后输出的内容
                "full_cot": "Complete chain of thought reasoning...", # 完整的思路链推理，包括问题
                "completion_tokens": 150,
                "reasoning_type": "step_by_step"
                }


    ├── incorrect_base_solution/ # 模型通过推理得出错误答案的问题
    ├── correct_base_solution_forced_answer/ # 正确答案的问题，最终答案被强制回答
    ├── incorrect_base_solution_forced_answer/ # 错误答案问题，最终答案被强制填写
    └── correct_base_solution_sanity/ # 理智检查题目，答案正确（ 注意： 这些题目可以忽略）

├── deepseek-r1-distill-qwen-14b/
  └── temperature_0.6_top_p_0.95/
    ├── correct_base_solution/
    ├── incorrect_base_solution/
    ├── correct_base_solution_forced_answer/
    ├── incorrect_base_solution_forced_answer/
    └── correct_base_solution_sanity/
```
