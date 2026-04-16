# results/ 目录说明

本目录存放**Exp 2/3 统计分析结果**。

## 生成脚本：`cross_domain_analysis.py`（Exp 2/3）

## 目录结构

```
results/
├── cross_domain_methods_results.json   ← Exp 2 + Exp 3 全部统计结果
├── simple_results.md                   ← 关键发现的人工摘要（快速参考）
```

---

## `cross_domain_methods_results.json`

**生成**：运行根目录下的 `cross_domain_analysis.py`  
**内容**：Exp 2（跨域比较）和 Exp 3（跨方法一致性）的全部统计指标，JSON 格式

### 顶层结构

| 键      | 对应实验           | 内容                                 |
| ------- | ------------------ | ------------------------------------ |
| `exp2a` | Exp 2a — 类别分布  | Gini 系数 + JSD                      |
| `exp2b` | Exp 2b — 注意力熵  | Receiver head score 分布熵           |
| `exp2c` | Exp 2c — 因果拓扑  | Causal Reach Ratio + Anchor 位置分布 |
| `exp3`  | Exp 3 — 方法一致性 | Kendall's W / τ / Top-k 重叠率       |

### `exp2a`

```
exp2a
├── gini
│   ├── gpqa / math
│   │   ├── values: list[float]   # 每道题的 Gini 系数（20 个值）
│   │   ├── mean: float
│   │   └── ci95: [lo, hi]        # Bootstrap 95% CI
│   ├── permutation_p: float      # 置换检验 p 值（双尾，10000次）
│   ├── cliffs_delta: float       # Cliff's δ 效应量
│   └── effect_size: str          # "negligible" / "small" / "medium" / "large"
└── jsd
    ├── value: float              # Jensen-Shannon Divergence（log2，[0,1]）
    ├── ci95: [lo, hi]            # Bootstrap CI（对 chunk 标签重采样，5000次）
    ├── label_freq_gpqa: dict     # 8类标签在 GPQA 中的频率
    └── label_freq_math: dict     # 8类标签在 Math 中的频率
```

**当前结果**：Gini 无显著差异（p=0.48，δ=0.11）；JSD=0.095，主要差异来自 `plan_generation`（Math 11.6% vs GPQA 1.1%）和 `uncertainty_management`（GPQA 29.0% vs Math 11.8%）。

### `exp2b`

```
exp2b
├── gpqa / math
│   ├── values: list[float]       # 每道题的注意力熵（20 个值）
│   ├── mean: float
│   └── ci95: [lo, hi]
├── permutation_p, cliffs_delta, effect_size
└── interpretation: str           # 指标说明（代理指标注记）
```

**指标说明**：以每道题内 `receiver_head_score` 的 Shannon 熵作为注意力集中度代理。高熵=多播（多句子均等重要），低熵=单跳（少数句子主导）。  
**当前结果**：GPQA 熵略高（6.24 vs 6.05），但差异不显著（p=0.35，δ=0.13）。

### `exp2c`

```
exp2c
├── causal_reach_ratio
│   ├── gpqa / math
│   │   ├── values: list[float]   # 每道题的远程/近程影响比（20 个值）
│   │   ├── mean: float
│   │   └── ci95: [lo, hi]
│   └── permutation_p, cliffs_delta, effect_size
└── anchor_location
    ├── gpqa / math
    │   ├── n_anchors: int        # 高重要性句子数（z-score > 1.0）
    │   ├── mean_pos: float       # Anchor 归一化位置均值（0=链首，1=链尾）
    │   └── values: list[float]   # 每个 anchor 的归一化位置
    ├── mann_whitney_u, mann_whitney_p
    ├── cliffs_delta, effect_size
```

**当前结果**：

- Causal Reach Ratio 无显著差异（p=0.47）
- **Anchor 位置**：GPQA anchor 出现更晚（mean=0.36 vs Math 0.28），p=0.0009，δ=0.18（small）——这是所有 Exp 2 指标中最显著的跨域差异。

### `exp3`

```
exp3
├── gpqa / math
│   ├── domain: str
│   ├── n_problems_used: int      # 三方法均有数据的题目数
│   ├── kendalls_W
│   │   ├── values: list[float]   # 每道题的 W 值
│   │   ├── mean: float
│   │   └── ci95: [lo, hi]
│   ├── pairwise_tau
│   │   ├── m1_m2 / m1_m3 / m2_m3
│   │   │   ├── mean: float       # 每道题 Kendall's τ 的均值
│   │   │   ├── values: list[float]
│   │   │   └── ci95: [lo, hi]
│   └── topk_agreement
│       └── "3" / "5" / "10"
│           ├── 12 / 13 / 23: float   # 两两 top-k 集合重叠率
│           └── 3way: float           # 三方均在 top-k 的比例
└── cross_domain_W
    ├── mann_whitney_u, mann_whitney_p
    ├── cliffs_delta, effect_size
    ├── mean_W_gpqa, mean_W_math
    └── ci_diff_approx: [lo, hi]
```

**方法映射**：`m1`=方法一（CF-KL），`m2`=方法二（receiver head score），`m3`=方法三（causal masking 列均值）

**当前结果**：

- Kendall's W 两域均约 0.29，跨方法整体一致性有限
- 方法一与方法二正相关（τ≈+0.14），方法一与方法三负相关（τ≈-0.13）
- 跨域 W 无显著差异（p=0.84）——方法一致性水平在两域相近

---

## `simple_results.md`

5 行人工摘要，快速浏览核心发现用。写论文时可直接参考。
