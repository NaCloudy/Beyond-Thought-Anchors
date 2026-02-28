# Thought Anchors ⚓ — 代码库中文说明

## 一、项目概述

本项目研究**推理型大语言模型 (LLM) 的 Chain-of-Thought (CoT) 推理过程**，通过三种方法识别推理链中的关键步骤（**Thought Anchors / 思维锚点**）：

1. **黑盒方法 (Black-box)**：通过 API 调用，遮蔽 CoT 中的某个句子后重新生成，用 KL 散度和准确率变化衡量该句子的重要性
2. **注意力方法 (Attention-based)**：分析模型的注意力模式，找到 "receiver heads"（接收头）等关键结构
3. **因果方法 (Causal)**：在注意力层直接遮蔽 token，对比 logit 分布变化

---

## 二、文件夹结构详解

```
thought-anchors/
│
├── 🔧 根目录脚本（黑盒分析主流程）
│   ├── generate_rollouts.py      # 步骤1: 生成推理 rollout 数据
│   ├── analyze_rollouts.py       # 步骤2: 分析 rollout 并计算重要性指标
│   ├── step_attribution.py       # 步骤3: 计算句子到句子的因果重要性分数
│   ├── plots.py                  # 步骤4: 生成论文中的图表
│   ├── sentence_scatter_and_ttests.py  # 统计检验（配对t检验 + 散点图）
│   ├── prompts.py                # LLM自动标注提示词（句子分类标签）
│   ├── utils.py                  # 通用工具函数（答案提取、文本分块、数据集加载等）
│   ├── selected_problems.json    # MATH数据集精选题目（25%-75%准确率范围的挑战性问题）
│   └── requirements.txt          # Python 依赖
│
├── 📊 masking_graphs/            # MMLU数据集上的遮蔽因果图实验
│   ├── base_responses.py         # 非思考模式基线回答生成（Fireworks API）
│   ├── base_responses_thinking.py # 思考模式回答生成（提取思考内容 + 最终答案）
│   ├── analyze_thinking.py       # thinking vs non-thinking 模式对比分析
│   ├── generate_graph_data.py    # 核心: 生成句子级KL散度因果图数据（SuppObj）
│   ├── graph_funcs.py            # 图分析函数（将KL矩阵建模为有向图，计算图指标）
│   ├── utils.py                  # 文本块对齐工具
│   ├── utils_sentences.py        # 句子分割工具
│   ├── visualization_utils.py    # Plotly交互式可视化
│   ├── constants.py              # MMLU学科分类常量 + 颜色映射
│   ├── download_mmlu.py          # 下载MMLU数据集
│   ├── plot_*.py                 # 各种绘图脚本
│   ├── printDomainDF.py          # 打印领域统计DataFrame
│   └── resample/                 # 重采样基础设施
│       ├── supp.py               # 核心类 SuppObj: 句子遮蔽 → 重采样 → KL矩阵
│       ├── rollouts.py           # LLM API 响应数据类（FwResponse, Rollouts 等）
│       ├── kl_funcs.py           # KL散度计算函数
│       ├── sentence_splitter.py  # 句子分割器
│       ├── fireworks_logprobs.py # Fireworks API logprobs获取
│       ├── together_logprobs.py  # Together API logprobs获取
│       ├── provider_config.py    # API提供商配置
│       ├── rate_limiter.py       # API速率限制器
│       ├── map_to_openrouter.py  # OpenRouter模型映射
│       ├── openrouter_clean.py   # OpenRouter API 调用工具
│       ├── compare_suppobj.py    # 比较不同 SuppObj
│       └── plot_suppression.py   # 抑制效果绘图
│
├── 🔬 whitebox-analyses/         # 白盒注意力分析（需要本地加载模型）
│   ├── README_on_pkld_decorator.md  # pkld缓存装饰器说明
│   ├── pytorch_models/           # PyTorch模型工具
│   │   ├── model_loader.py       # 模型加载器（ModelLoader类，支持缓存、多精度）
│   │   ├── model_config.py       # 模型配置
│   │   ├── analysis.py           # 注意力权重 + logits 提取（前向传播分析）
│   │   ├── hooks.py              # 注意力遮蔽 hook 管理（HookManager类）
│   │   ├── ablation.py           # 消融实验
│   │   ├── common.py             # 通用工具
│   │   └── rope_utils.py         # RoPE位置编码工具
│   ├── attention_analysis/       # 注意力模式分析
│   │   ├── attn_funcs.py         # 注意力矩阵提取和分析
│   │   ├── attn_supp_funcs.py    # 白盒版句子级KL抑制矩阵（注意力层直接遮蔽）
│   │   ├── logits_funcs.py       # Logits处理函数
│   │   ├── receiver_head_funcs.py # Receiver Head 识别和分析
│   │   └── tokenizer_funcs.py    # Tokenizer工具
│   └── scripts/                  # 白盒实验执行脚本
│       ├── prep_attn_cache.py     # 预计算注意力缓存
│       ├── prep_suppression_mtxs.py # 预计算抑制KL矩阵
│       ├── plot_attention_heads.py  # 绘制注意力头
│       ├── plot_one_attn_matrix.py  # 绘制单个注意力矩阵
│       ├── plot_many_attn_matrices.py # 批量绘制注意力矩阵
│       ├── plot_suppression_matrix.py # 绘制抑制矩阵
│       ├── plot_rec_taxonomy.py     # Receiver Head分类图
│       ├── plot_kurt_stats.py       # 峰度统计图
│       ├── eval_receiver_head_*.py  # Receiver Head评估脚本
│       ├── generate_rec_csvs.py     # 生成Receiver Head CSV
│       └── print_case_study_transcript.py # 打印案例分析
│
├── 🧪 misc-experiments/          # 辅助实验
│   ├── generate_cots.py          # 通过OpenRouter API生成CoT（支持多模型）
│   ├── kl_attribution.py         # 本地模型KL归因（支持4-bit量化）
│   ├── analyze_step_transitions.py # 步骤转换分析
│   ├── attribution_benchmark.py  # 归因基准测试
│   └── get_stats.ipynb           # 统计分析notebook
│
├── 📤 misc-scripts/
│   └── push_hf_dataset.py        # 上传数据集到HuggingFace
│
└── 📁 analysis/                  # 分析输出结果
    └── basic/
        └── deepseek-r1-distill-qwen-14b/
            ├── alpha_0.5/           # Laplace平滑 α=0.5
            ├── alpha_1.0/           # Laplace平滑 α=1.0
            ├── alpha_1e-09/         # Laplace平滑 α≈0（最小平滑）
            ├── correct_base_solution/   # 正确基线解的分析
            └── incorrect_base_solution/ # 错误基线解的分析
```

---

## 三、核心概念说明

### 3.1 什么是 "Thought Anchor"（思维锚点）？

LLM 在 Chain-of-Thought 推理时，某些句子对后续推理有**不成比例的巨大影响**。这些句子通常是：

- **规划句 (Plan Generation)**：制定解题策略
- **回溯句 (Uncertainty Management / Backtracking)**：发现错误后改变方向

### 3.2 三种重要性度量 (Importance Metrics)

| 指标                          | 方法 | 说明                                                                                             |
| ----------------------------- | ---- | ------------------------------------------------------------------------------------------------ |
| **Resampling Importance**     | 黑盒 | 从句子 i 处截断并重新生成，对比保留 vs 不保留该句子的准确率/KL散度差异                           |
| **Counterfactual Importance** | 黑盒 | 在重采样的 rollout 中找到与目标句子语义相似的句子，比较包含/不包含某句子时这些相似句子出现的概率 |
| **Forced Answer Importance**  | 黑盒 | 删除某句子后直接强制模型输出最终答案，衡量答案准确率变化                                         |

### 3.3 句子分类标签 (Function Tags)

每个推理句子被 LLM 自动标注为以下类别之一：

- **Active Computation** (主动计算)
- **Fact Retrieval** (事实回忆)
- **Plan Generation** (规划)
- **Uncertainty Management** (不确定性管理 / 回溯)
- **Result Consolidation** (结果整合)
- **Self Checking** (自检)
- **Problem Setup** (问题设定)
- **Final Answer Emission** (最终答案)

---

## 四、复现黑盒方法 (Black-box Analysis) 的完整步骤

### 前置条件

1. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

2. **配置 API 密钥**（创建 `.env` 文件）

   ```bash
   # 至少需要以下之一用于生成 rollout：
   NOVITA_API_KEY=your_key        # Novita API（默认 provider）
   TOGETHER_API_KEY=your_key      # Together API
   FIREWORKS_API_KEY=your_key     # Fireworks API

   # 用于句子自动标注（analyze_rollouts.py 需要）：
   OPENAI_API_KEY=your_key        # OpenAI API（用于GPT-4标注句子类别）
   ```

   > **注意**：也可以使用 `--provider Local` 在本地运行模型（需要 GPU），此时不需要 API Key。

3. **准备数据**（二选一）
   - **方式A**：下载现成的 rollout 数据集
     ```bash
     # 从 HuggingFace 下载
     # https://huggingface.co/datasets/uzaymacar/math-rollouts
     ```
   - **方式B**：自己生成数据（见下面步骤1）

### 步骤1: 生成推理 Rollout (`generate_rollouts.py`)

为每个数学问题生成 base solution（基线解），然后对每个句子（chunk）进行遮蔽重采样。

```bash
# 生成正确基线解的 rollout（使用 Novita API）
python generate_rollouts.py \
    -m "deepseek/deepseek-r1-distill-qwen-14b" \
    -b correct \
    -r default \
    -o math_rollouts \
    -np 100 \          # 100个问题
    -nr 100 \          # 每个chunk 100次rollout
    -t 0.6 \           # temperature
    -tp 0.95 \         # top_p
    -p Novita           # API provider

# 生成错误基线解的 rollout
python generate_rollouts.py \
    -m "deepseek/deepseek-r1-distill-qwen-14b" \
    -b incorrect \
    -r default \
    -o math_rollouts \
    -np 100 -nr 100 -t 0.6 -tp 0.95 -p Novita

# 生成 forced_answer rollout（强制回答模式）
python generate_rollouts.py \
    -m "deepseek/deepseek-r1-distill-qwen-14b" \
    -b correct \
    -r forced_answer \
    -o math_rollouts \
    -np 100 -nr 100 -t 0.6 -tp 0.95 -p Novita

# 也支持本地模型运行（需要 GPU）：
python generate_rollouts.py -m "deepseek/deepseek-r1-distill-qwen-14b" -b correct -p Local -q  # -q 启用4bit量化
```

**输出结构**：

```
math_rollouts/
└── deepseek-r1-distill-qwen-14b/
    └── temperature_0.6_top_p_0.95/
        ├── correct_base_solution/
        │   ├── problem_0/
        │   │   ├── problem.json          # 原始数学问题
        │   │   ├── base_solution.json     # 基线解答
        │   │   ├── chunks.json            # 分块后的句子
        │   │   ├── chunk_0/solutions.json # chunk_0 遮蔽后的 100 次 rollout
        │   │   ├── chunk_1/solutions.json # chunk_1 遮蔽后的 100 次 rollout
        │   │   └── ...
        │   └── problem_1/
        │       └── ...
        ├── incorrect_base_solution/
        └── correct_base_solution_forced_answer/
```

### 步骤2: 分析 Rollout 并计算重要性指标 (`analyze_rollouts.py`)

处理生成的 rollout 数据，计算三种重要性指标，并用 GPT-4 对句子自动分类标注。

```bash
python analyze_rollouts.py \
    -ic "math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution" \
    -ii "math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution" \
    -icf "math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution_forced_answer" \
    -iif "math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution_forced_answer" \
    -o "analysis/basic/deepseek-r1-distill-qwen-14b/alpha_1.0" \
    --laplace_alpha 1.0 \
    -im "counterfactual_importance_accuracy"
```

**此步骤会**：

- 为每个问题生成 `chunks_labeled.json`（含句子文本、类别标签、重要性分数）
- 计算 `resampling_importance`（重采样重要性）
- 计算 `counterfactual_importance`（反事实重要性）
- 计算 `forced_importance`（强制回答重要性）
- 方差分析输出到 `variance_analysis/` 目录

### 步骤3: 句子到句子归因 (`step_attribution.py`)

计算每对句子 (i→j) 之间的因果重要性。详细原理见下方 [附录A](#附录astep_attributionpy-详解)。

```bash
python step_attribution.py \
    -ad "math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95" \
    -od "analysis/step_attribution" \
    -st 0.8 \           # 语义相似度阈值
    -co                  # 仅分析正确基线解
    # 可选参数：
    # -mc 50             # 最大chunk数
    # -mp 10             # 最大问题数
    # -ip "0,5,10"       # 仅分析指定问题
    # -ex "SC,PS"        # 排除 Self Checking 和 Problem Setup 类别
    # -nt 15             # 仅展示重要性最高的15个句子
```

### 步骤4: 生成图表 (`plots.py`)

```bash
python plots.py -m qwen-14b --normalize
```

### 步骤5（可选）: 统计检验 (`sentence_scatter_and_ttests.py`)

```bash
python sentence_scatter_and_ttests.py
```

---

## 五、关键注意事项

### 5.1 API 成本估算

- 每个问题 × 每个 chunk × 100 次 rollout = **大量 API 调用**
- 以 100 个问题、平均 15 个 chunk 为例：100 × 15 × 100 = **150,000 次 API 调用**
- 建议先用少量问题 (`-np 5 -nr 10`) 进行测试

### 5.2 本地运行替代方案

- 可使用 `--provider Local` 在本地 GPU 上运行模型
- 支持 4-bit 量化 (`-q`)，可降低显存需求
- DeepSeek-R1-Distill-Qwen-14B 全精度需要 ~28GB 显存，4-bit 量化约需 ~8GB

### 5.3 数据集快捷方式

如果不想自己生成 rollout，可以直接从 HuggingFace 下载预生成的数据集：

```python
from datasets import load_dataset
dataset = load_dataset("uzaymacar/math-rollouts")
```

然后跳过步骤1，直接执行步骤2-5。

### 5.4 `pkld` 缓存装饰器

代码库中大量使用 `@pkld` 装饰器自动缓存函数结果到磁盘，避免重复的 API 调用和模型推理。参见 `whitebox-analyses/README_on_pkld_decorator.md`。

---

## 附录A：`step_attribution.py` 详解

### A.1 文件定位与目标

`step_attribution.py` 是黑盒分析流程中的 **步骤3**，在 `analyze_rollouts.py` 计算出每个句子的单维重要性分数之后，进一步计算**句子与句子之间的因果关系矩阵**——即"句子 i 对句子 j 的出现有多大影响"。

> 和步骤2的区别：步骤2回答"哪个句子重要"，步骤3回答"**谁影响了谁**"。

### A.2 核心算法原理

对于一条包含 $n$ 个句子的推理链，脚本计算一个 $n \times n$ 的上三角重要性矩阵 $M$，其中 $M[i,j]$（$j > i$）表示**句子 i 对句子 j 的因果重要性**。

**计算 $M[i,j]$ 的流程**：

```
对于每一对 (i, j)，其中 j > i：

1. 收集"保留句子 i"的 rollout：
   - 从 chunk_{i+1}/ 加载（从句子 i+1 处开始重采样 → 句子 i 被保留）

2. 收集"移除句子 i"的 rollout：
   - 从 chunk_{i}/ 加载（从句子 i 处开始重采样 → 句子 i 被移除）

3. 在每组 rollout 中，用语义相似度寻找"句子 j 的对应物"：
   - 将句子 j 的 embedding 与 rollout 中所有句子的 embedding 计算余弦相似度
   - 找到每个 rollout 中与句子 j 最匹配的句子（相似度 ≥ 阈值，默认 0.8）

4. 计算匹配率差异：
   M[i,j] = match_rate(保留i) - match_rate(移除i)
```

**直觉解释**：如果保留句子 i 时，下游 rollout 有 80% 概率生成类似句子 j 的内容，而移除句子 i 后只有 30%，则 $M[i,j] = 0.5$，说明句子 i 的存在强烈促进了句子 j 的出现。

### A.3 代码结构与关键函数

```
step_attribution.py (993 行)
│
├── 数据加载
│   ├── load_problem()              # 加载 problem.json
│   ├── load_base_solution()        # 加载 base_solution.json
│   ├── load_chunks()               # 加载 chunks_labeled.json（带标签的句子）
│   └── load_chunk_rollouts()       # 加载 chunk_N/solutions.json（N个rollout）
│
├── 语义匹配（核心）
│   ├── extract_steps_from_rollout()           # 将 rollout 文本分割成句子列表
│   ├── compute_embedding_similarity()         # 计算两段文本的余弦相似度
│   └── find_best_matches_fully_batched()      # 批量匹配：在所有rollout中找句子j的最佳对应
│       └── 使用 SentenceTransformer('all-MiniLM-L6-v2') 编码
│       └── sklearn.cosine_similarity 计算相似度
│       └── 超过阈值(默认0.8)才算"匹配成功"
│
├── 重要性矩阵计算
│   ├── process_chunk_pair()                   # 计算单个 (i,j) 对的重要性（支持多进程）
│   └── compute_step_importance_matrix()       # 计算完整 n×n 矩阵（带缓存）
│       └── 预加载所有 rollout + 预计算所有 embedding（避免重复IO和推理）
│       └── 向量化计算 cosine_similarity（高效批处理）
│       └── 缓存结果到 .npz 文件
│
├── 句子筛选与过滤
│   ├── get_function_tag_prefix()              # 提取句子类别缩写（如 AC, PG, UM）
│   ├── get_category_from_abbreviation()       # 缩写 → 完整类别名映射
│   ├── filter_chunks_by_excluded_tags()       # 按类别过滤句子
│   ├── calculate_sentence_importance_scores() # 计算综合重要性 = 出度 + 入度
│   └── select_top_sentences()                 # 选取重要性最高的N个句子
│
├── 可视化输出
│   ├── plot_step_importance_matrix()          # 热力图（RdBu色系，上三角遮蔽）
│   ├── outgoing_importance 柱状图             # 每个句子对下游的平均影响
│   └── incoming_importance 柱状图             # 每个句子受上游的平均影响
│
├── 辅助函数
│   ├── get_problem_dirs()                     # 扫描目录，筛选数据完整的problem
│   └── analyze_step_attribution()             # 主调度: 遍历问题 → 计算矩阵 → 画图 → 存JSON
│
└── main()                                     # CLI入口，解析参数并启动分析
```

### A.4 输入依赖

脚本需要步骤1和步骤2的产出：

| 输入文件                           | 来源                   | 说明                              |
| ---------------------------------- | ---------------------- | --------------------------------- |
| `problem_N/problem.json`           | `generate_rollouts.py` | 原始数学问题                      |
| `problem_N/base_solution.json`     | `generate_rollouts.py` | 基线推理解答                      |
| `problem_N/chunks_labeled.json`    | `analyze_rollouts.py`  | 带标签的句子列表                  |
| `problem_N/chunk_K/solutions.json` | `generate_rollouts.py` | 在第K个句子处截断后的100次rollout |

### A.5 输出文件

```
analysis/step_attribution/{model_name}/
├── correct_base_solution/
│   └── problem_N/
│       ├── base_solution.json           # 复制的基线解答
│       ├── chunk_texts.json             # 句子列表
│       ├── step_importance.json         # 每个句子对下游的详细影响分数
│       ├── summary.json                 # 汇总统计（最有影响力/最受影响的句子）
│       ├── cache/
│       │   └── step_importance_matrix_t0.8.npz  # 缓存的numpy矩阵
│       └── plots/
│           ├── step_importance_matrix.png   # n×n 热力图
│           ├── outgoing_importance.png      # 出度柱状图
│           └── incoming_importance.png      # 入度柱状图
└── incorrect_base_solution/
    └── ...（同上）
```

### A.6 关键参数说明

| 参数                             | 默认值                                         | 说明                                 |
| -------------------------------- | ---------------------------------------------- | ------------------------------------ |
| `-ad` / `--analysis_dir`         | `math_rollouts/.../temperature_0.6_top_p_0.95` | rollout 数据目录                     |
| `-od` / `--output_dir`           | `analysis/step_attribution`                    | 输出目录                             |
| `-st` / `--similarity_threshold` | `0.8`                                          | 语义相似度阈值。越高越严格，匹配越少 |
| `-mc` / `--max_chunks`           | `None` (不限)                                  | 每个问题最多分析多少个句子           |
| `-mp` / `--max_problems`         | `None` (不限)                                  | 最多分析多少个问题                   |
| `-co` / `--correct_only`         | `False` (分析incorrect)                        | 是否仅分析正确基线解                 |
| `-ex` / `--exclude_tags`         | `None`                                         | 排除的类别缩写，如 `"SC,PS,FAE"`     |
| `-nt` / `--num_top_sentences`    | `None` (全部)                                  | 仅展示重要性最高的N个句子            |

### A.7 性能优化设计

1. **预计算所有 embedding**：一次性编码所有 rollout 的所有句子，避免重复调用 SentenceTransformer
2. **向量化 cosine_similarity**：利用 sklearn 的矩阵运算，而非逐对计算
3. **NPZ 缓存**：计算结果持久化到 `.npz` 文件，重复运行时直接加载
4. **预加载所有 rollout**：一次性从磁盘读取所有 `solutions.json`，避免在内层循环中反复IO
