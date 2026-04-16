"""
cross_domain_analysis.py
========================
Exp 2 (Cross-Domain Comparison) + Exp 3 (Method Agreement) 分析脚本

输入数据：
  - 方法一（黑盒重采样）：analysis/{domain}/correct_base_solution/analysis_results.json
  - 方法二（Receiver Head）：csvs/{domain}/receiver_head_scores_correct_qwen-14b_k32_pi16.csv
  - 方法三（因果遮蔽）：kl_results/{domain}/qwen-14b/correct/problem_{N}_kl.npy

输出：results/cross_domain_results.json  +  控制台打印摘要
"""

import csv
import json
import os
import sys
import hashlib
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.stats import kendalltau, mannwhitneyu

# ──────────────────────────────────────────────────────────────────────────────
# 路径配置（相对于本脚本所在目录）
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

DOMAINS = ["gpqa", "math"]
MODEL_NAME = "qwen-14b"
REC_CSV_SUFFIX = f"receiver_head_scores_correct_{MODEL_NAME}_k32_pi16.csv"

SENTINEL = -20.72326584  # KL 矩阵上三角填充的哨兵值


# ──────────────────────────────────────────────────────────────────────────────
# 1. 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def load_method1(domain: str) -> list[dict]:
    """
    返回 per-problem 数据列表，每项：
      {
        'problem_id': str,
        'scores': list[float],   # counterfactual_importance_kl，每个 chunk 一个
        'tags':   list[list[str]],
        'n': int
      }
    """
    path = os.path.join(BASE_DIR, "analysis", domain,
                        "correct_base_solution", "analysis_results.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    problems = []
    for entry in raw:
        pid = str(entry["problem_idx"])
        chunks = entry["labeled_chunks"]
        scores = [c["counterfactual_importance_kl"] for c in chunks]
        tags = [c.get("function_tags", []) for c in chunks]
        problems.append({"problem_id": pid, "scores": scores, "tags": tags, "n": len(chunks)})
    return problems


def load_method2(domain: str) -> dict[str, list[float]]:
    """
    返回 {problem_id: [receiver_head_score, ...]} （按 sentence_idx 排序）
    """
    path = os.path.join(BASE_DIR, "csvs", domain, REC_CSV_SUFFIX)
    result: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(row["problem_number"])
            sidx = int(row["sentence_idx"])
            raw = row["receiver_head_score"]
            score = float(raw) if raw.strip() else 0.0  # post-convergence 行为空
            result[pid].append((sidx, score))

    # 按 sentence_idx 排序，返回纯分数列表
    return {pid: [s for _, s in sorted(v)] for pid, v in result.items()}


def load_method3(domain: str) -> dict[str, np.ndarray]:
    """
    返回 {problem_id: per_sentence_causal_score (1D array)}

    per-sentence score = 该句对所有后续句 KL 影响的均值
    （即 KL 矩阵第 i 列中 j > i 部分的均值；哨兵值排除）
    """
    kl_dir = os.path.join(BASE_DIR, "kl_results", domain, MODEL_NAME, "correct")
    result = {}
    for fname in os.listdir(kl_dir):
        if not fname.endswith("_kl.npy"):
            continue
        pid = fname.replace("problem_", "").replace("_kl.npy", "")
        mat = np.load(os.path.join(kl_dir, fname))
        N = mat.shape[0]
        scores = []
        for i in range(N):
            col = mat[i + 1:, i]  # 第 i 列，仅 j > i 部分
            # 排除哨兵值（上三角填充）
            valid = col[col > SENTINEL + 1]
            scores.append(float(valid.mean()) if len(valid) > 0 else 0.0)
        result[pid] = np.array(scores)
    return result


def load_method3_matrix(domain: str) -> dict[str, np.ndarray]:
    """返回 {problem_id: raw KL matrix (N×N)}，用于 Exp 2c 因果拓扑分析"""
    kl_dir = os.path.join(BASE_DIR, "kl_results", domain, MODEL_NAME, "correct")
    result = {}
    for fname in os.listdir(kl_dir):
        if not fname.endswith("_kl.npy"):
            continue
        pid = fname.replace("problem_", "").replace("_kl.npy", "")
        result[pid] = np.load(os.path.join(kl_dir, fname))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. 统计工具函数
# ──────────────────────────────────────────────────────────────────────────────

def gini(scores: list[float]) -> float:
    """计算 Gini 系数（0 = 完全均等，1 = 完全集中）"""
    arr = np.array(scores, dtype=float)
    arr = arr - arr.min()  # 确保非负
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr)) / (n * arr.sum()) - (n + 1) / n)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Divergence（log2，范围 [0,1]）"""
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    # 避免 log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log2(p / m), 0.0)
        kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0)
    return float(0.5 * kl_pm.sum() + 0.5 * kl_qm.sum())


def permutation_test_2samp(a: np.ndarray, b: np.ndarray,
                            n_perm: int = 10000) -> float:
    """双样本置换检验，返回 p 值（双尾）"""
    a, b = np.array(a), np.array(b)
    obs = abs(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    na = len(a)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        diff = abs(perm[:na].mean() - perm[na:].mean())
        if diff >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta（非参数效应量，范围 [-1, 1]）"""
    a, b = np.array(a), np.array(b)
    n = len(a) * len(b)
    dom = sum(1 for x in a for y in b if x > y) - sum(1 for x in a for y in b if x < y)
    return dom / n


def bootstrap_ci(values: np.ndarray, func=np.mean,
                 n_boot: int = 5000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap 置信区间"""
    rng = np.random.default_rng(42)
    boot_stats = [func(rng.choice(values, size=len(values), replace=True))
                  for _ in range(n_boot)]
    lo = (1 - ci) / 2
    hi = 1 - lo
    return float(np.quantile(boot_stats, lo)), float(np.quantile(boot_stats, hi))


def bootstrap_jsd_ci(tags_a: list[str], tags_b: list[str],
                     all_labels: list[str], n_boot: int = 5000) -> tuple[float, float]:
    """对 JSD 做 bootstrap CI（对 chunk 标签列表重采样）"""
    rng = np.random.default_rng(42)
    jsds = []
    for _ in range(n_boot):
        sa = rng.choice(tags_a, size=len(tags_a), replace=True)
        sb = rng.choice(tags_b, size=len(tags_b), replace=True)
        pa = np.array([np.sum(sa == lab) for lab in all_labels], dtype=float) + 1e-9
        pb = np.array([np.sum(sb == lab) for lab in all_labels], dtype=float) + 1e-9
        jsds.append(jsd(pa, pb))
    return float(np.quantile(jsds, 0.025)), float(np.quantile(jsds, 0.975))


def cliffs_delta_interp(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def shannon_entropy(dist: np.ndarray) -> float:
    """Shannon 熵（以 bit 为单位）"""
    dist = np.array(dist, dtype=float)
    dist = dist[dist > 0]
    if len(dist) == 0:
        return 0.0
    p = dist / dist.sum()
    return float(-np.sum(p * np.log2(p)))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Exp 2a — 类别分布（Gini + JSD）
# ──────────────────────────────────────────────────────────────────────────────

ALL_LABELS = ["problem_setup", "plan_generation", "active_computation",
              "fact_retrieval", "uncertainty_management",
              "result_consolidation", "self_checking", "final_answer_emission"]


def normalize_tag(tag: str) -> str:
    """统一标签格式（下划线分隔）"""
    return tag.lower().replace(" ", "_").replace("-", "_")


def exp2a_gini_jsd(m1_gpqa: list, m1_math: list) -> dict:
    """Exp 2a: Gini 系数 + JSD"""

    # ── Gini per problem ──
    gini_gpqa = [gini(p["scores"]) for p in m1_gpqa]
    gini_math = [gini(p["scores"]) for p in m1_math]

    p_gini = permutation_test_2samp(np.array(gini_gpqa), np.array(gini_math))
    d_gini = cliffs_delta(np.array(gini_gpqa), np.array(gini_math))
    ci_gini_gpqa = bootstrap_ci(np.array(gini_gpqa))
    ci_gini_math = bootstrap_ci(np.array(gini_math))

    # ── JSD on label distributions ──
    all_tags_gpqa = [normalize_tag(t)
                     for p in m1_gpqa for tags in p["tags"] for t in tags]
    all_tags_math = [normalize_tag(t)
                     for p in m1_math for tags in p["tags"] for t in tags]

    all_labels_norm = [normalize_tag(l) for l in ALL_LABELS]

    pa = np.array([all_tags_gpqa.count(lab) for lab in all_labels_norm], dtype=float) + 1e-9
    pm = np.array([all_tags_math.count(lab) for lab in all_labels_norm], dtype=float) + 1e-9
    jsd_val = jsd(pa, pm)
    ci_jsd = bootstrap_jsd_ci(np.array(all_tags_gpqa), np.array(all_tags_math),
                               all_labels_norm)

    # ── Label frequency tables ──
    def label_freq(tag_list):
        total = len(tag_list) + 1e-9
        return {lab: round(tag_list.count(lab) / total, 4)
                for lab in all_labels_norm}

    return {
        "gini": {
            "gpqa": {"values": gini_gpqa, "mean": float(np.mean(gini_gpqa)),
                     "ci95": ci_gini_gpqa},
            "math": {"values": gini_math, "mean": float(np.mean(gini_math)),
                     "ci95": ci_gini_math},
            "permutation_p": p_gini,
            "cliffs_delta": d_gini,
            "effect_size": cliffs_delta_interp(d_gini),
        },
        "jsd": {
            "value": jsd_val,
            "ci95": ci_jsd,
            "label_freq_gpqa": label_freq(all_tags_gpqa),
            "label_freq_math": label_freq(all_tags_math),
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Exp 2b — 注意力熵（Receiver Head 分数分布的 Shannon 熵）
# ──────────────────────────────────────────────────────────────────────────────

def exp2b_attention_entropy(m2_gpqa: dict, m2_math: dict,
                             m1_gpqa: list, m1_math: list) -> dict:
    """
    Exp 2b：注意力熵

    代理指标：将 receiver_head_score 在问题内的分布视为"注意力分布"，
    计算其 Shannon 熵。高熵 = 多播（多个句子均等重要），
    低熵 = 单跳（少数句子集中吸引注意力）。

    Note: 若后续有原始注意力矩阵，可替换此计算。
    """
    def problem_entropy(scores: list[float]) -> float:
        arr = np.array(scores, dtype=float)
        arr = arr - arr.min() + 1e-9  # 非负化
        return shannon_entropy(arr)

    # 按 problem_id 对齐
    def entropies_for_domain(m2: dict, m1: list) -> list[float]:
        ents = []
        for p in m1:
            pid = p["problem_id"]
            if pid in m2:
                ents.append(problem_entropy(m2[pid]))
        return ents

    ents_gpqa = entropies_for_domain(m2_gpqa, m1_gpqa)
    ents_math = entropies_for_domain(m2_math, m1_math)

    p_val = permutation_test_2samp(np.array(ents_gpqa), np.array(ents_math))
    d_val = cliffs_delta(np.array(ents_gpqa), np.array(ents_math))
    ci_gpqa = bootstrap_ci(np.array(ents_gpqa))
    ci_math = bootstrap_ci(np.array(ents_math))

    return {
        "gpqa": {"values": ents_gpqa, "mean": float(np.mean(ents_gpqa)), "ci95": ci_gpqa},
        "math": {"values": ents_math, "mean": float(np.mean(ents_math)), "ci95": ci_math},
        "permutation_p": p_val,
        "cliffs_delta": d_val,
        "effect_size": cliffs_delta_interp(d_val),
        "interpretation": (
            "高熵=多播(GPQA预期高)，低熵=单跳(Math预期低)。"
            "此代理指标基于 receiver_head_score 分布；若需严格注意力矩阵熵，替换 m2 输入。"
        )
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Exp 2c — 因果拓扑（Causal Reach Ratio + Anchor 位置分布）
# ──────────────────────────────────────────────────────────────────────────────

def causal_reach_ratio(mat: np.ndarray, threshold_frac: float = 0.5) -> float:
    """
    因果到达比 = 远程影响总量 / 近程影响总量

    对 KL 矩阵下三角（i < j，即句子 i 对后续句子 j 的影响）：
    - 距离 d = j - i
    - 阈值 = ceil(链长 × threshold_frac)
    - 远程：d > threshold；近程：d <= threshold
    排除哨兵值，仅统计有效影响。
    """
    N = mat.shape[0]
    threshold = max(1, int(N * threshold_frac))

    far_sum = 0.0
    near_sum = 0.0

    for i in range(N - 1):
        for j in range(i + 1, N):
            val = mat[j, i]
            if val <= SENTINEL + 1:
                continue
            # 有效影响（log-KL），转为正数（取 exp 或直接用 shifted 值）
            effective = val - SENTINEL  # 相对强度（正值越大=影响越大）
            if j - i > threshold:
                far_sum += effective
            else:
                near_sum += effective

    if near_sum == 0:
        return float("inf")
    return float(far_sum / near_sum)


def anchor_locations(scores: np.ndarray, z_threshold: float = 1.0) -> list[float]:
    """
    识别 anchor 句子并返回其归一化位置 (position / chain_length)

    anchor 定义：z-score > z_threshold
    """
    if len(scores) == 0:
        return []
    mu, sigma = scores.mean(), scores.std()
    if sigma == 0:
        return []
    n = len(scores)
    return [i / n for i, s in enumerate(scores) if (s - mu) / sigma > z_threshold]


def exp2c_causal_topology(m1_gpqa: list, m1_math: list,
                           m3_mat_gpqa: dict, m3_mat_math: dict) -> dict:
    """Exp 2c: Causal Reach Ratio + Anchor 位置分布"""

    # ── Causal Reach Ratio ──
    def reach_ratios(m3_mats: dict, m1: list) -> list[float]:
        ratios = []
        for p in m1:
            pid = p["problem_id"]
            if pid in m3_mats:
                r = causal_reach_ratio(m3_mats[pid])
                ratios.append(r)
        return ratios

    rr_gpqa = reach_ratios(m3_mat_gpqa, m1_gpqa)
    rr_math = reach_ratios(m3_mat_math, m1_math)

    # 去除 inf（全部影响都是近程时出现）
    rr_gpqa_clean = [r for r in rr_gpqa if np.isfinite(r)]
    rr_math_clean = [r for r in rr_math if np.isfinite(r)]

    p_rr = permutation_test_2samp(np.array(rr_gpqa_clean), np.array(rr_math_clean))
    d_rr = cliffs_delta(np.array(rr_gpqa_clean), np.array(rr_math_clean))
    ci_rr_gpqa = bootstrap_ci(np.array(rr_gpqa_clean))
    ci_rr_math = bootstrap_ci(np.array(rr_math_clean))

    # ── Anchor 位置分布 ──
    def all_anchor_locs(m1: list) -> np.ndarray:
        locs = []
        for p in m1:
            locs.extend(anchor_locations(np.array(p["scores"])))
        return np.array(locs)

    locs_gpqa = all_anchor_locs(m1_gpqa)
    locs_math = all_anchor_locs(m1_math)

    # Mann-Whitney U
    if len(locs_gpqa) > 0 and len(locs_math) > 0:
        mw_stat, mw_p = mannwhitneyu(locs_gpqa, locs_math, alternative="two-sided")
        d_loc = cliffs_delta(locs_gpqa, locs_math)
    else:
        mw_stat, mw_p, d_loc = None, None, None

    return {
        "causal_reach_ratio": {
            "gpqa": {"values": rr_gpqa_clean, "mean": float(np.mean(rr_gpqa_clean)) if rr_gpqa_clean else None, "ci95": ci_rr_gpqa},
            "math": {"values": rr_math_clean, "mean": float(np.mean(rr_math_clean)) if rr_math_clean else None, "ci95": ci_rr_math},
            "permutation_p": p_rr,
            "cliffs_delta": d_rr,
            "effect_size": cliffs_delta_interp(d_rr),
        },
        "anchor_location": {
            "gpqa": {
                "n_anchors": len(locs_gpqa),
                "mean_pos": float(locs_gpqa.mean()) if len(locs_gpqa) > 0 else None,
                "values": locs_gpqa.tolist(),
            },
            "math": {
                "n_anchors": len(locs_math),
                "mean_pos": float(locs_math.mean()) if len(locs_math) > 0 else None,
                "values": locs_math.tolist(),
            },
            "mann_whitney_u": mw_stat,
            "mann_whitney_p": float(mw_p) if mw_p is not None else None,
            "cliffs_delta": d_loc,
            "effect_size": cliffs_delta_interp(d_loc) if d_loc is not None else None,
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Exp 3 — 跨方法一致性
# ──────────────────────────────────────────────────────────────────────────────

def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    """分数转排名（高分=低排名=1）"""
    return stats.rankdata(-scores)


def kendalls_w(rankings: np.ndarray) -> float:
    """
    Kendall's W（concordance coefficient）
    rankings: shape (k, n) — k 个方法，n 个 items
    """
    k, n = rankings.shape
    mean_rank = (n + 1) / 2
    # 每列排名之和
    col_sums = rankings.sum(axis=0)
    S = np.sum((col_sums - k * mean_rank) ** 2)
    W = 12 * S / (k ** 2 * (n ** 3 - n))
    return float(np.clip(W, 0, 1))


def pairwise_kendall_tau(r1, r2, n_boot: int = 5000):
    """计算两个排名序列的 Kendall's τ + bootstrap CI"""
    tau, p = kendalltau(r1, r2)
    rng = np.random.default_rng(42)
    boot_taus = []
    n = len(r1)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        t, _ = kendalltau(np.array(r1)[idx], np.array(r2)[idx])
        if not np.isnan(t):
            boot_taus.append(t)
    ci = (float(np.quantile(boot_taus, 0.025)),
          float(np.quantile(boot_taus, 0.975))) if boot_taus else (None, None)
    return float(tau), float(p), ci


def topk_agreement(s1, s2, s3, k: int) -> dict:
    """
    Top-k 集合重叠率

    返回：
      pairwise_12, pairwise_13, pairwise_23: 两两重叠率
      three_way: 三方同时 top-k 的比例
    """
    n = len(s1)
    k = min(k, n)
    t1 = set(np.argsort(-np.array(s1))[:k])
    t2 = set(np.argsort(-np.array(s2))[:k])
    t3 = set(np.argsort(-np.array(s3))[:k])
    return {
        "pairwise_12": len(t1 & t2) / k,
        "pairwise_13": len(t1 & t3) / k,
        "pairwise_23": len(t2 & t3) / k,
        "three_way":   len(t1 & t2 & t3) / k,
    }


def align_three_methods(m1_problem: dict,
                         m2_all: dict,
                         m3_all: dict) -> tuple | None:
    """
    对齐三种方法在同一个问题上的分数，返回 (s1, s2, s3)

    对齐策略：
    - 方法一有 N 个 chunk
    - 方法二有 N-1 个句子（不含最后一句）
    - 方法三有 N-1 个 causal score（KL 矩阵大小 N-1）
    → 取前 min(len1-1, len2, len3) 个位置

    返回 None 如果任何方法数据缺失。
    """
    pid = m1_problem["problem_id"]
    s1 = np.array(m1_problem["scores"])

    if pid not in m2_all:
        return None
    s2 = np.array(m2_all[pid])

    if pid not in m3_all:
        return None
    s3 = m3_all[pid]

    # 对齐长度：方法一去掉最后一句（与方法二三对齐）
    n = min(len(s1) - 1, len(s2), len(s3))
    if n < 3:
        return None

    return s1[:n], s2[:n], s3[:n]


def exp3_method_agreement(m1: list, m2_all: dict, m3_all: dict,
                           domain_label: str) -> dict:
    """Exp 3：对单个域计算方法一致性指标"""
    Ws = []
    taus_12, taus_13, taus_23 = [], [], []
    topk_results = {k: {"12": [], "13": [], "23": [], "3way": []}
                    for k in [3, 5, 10]}
    n_valid = 0

    for prob in m1:
        aligned = align_three_methods(prob, m2_all, m3_all)
        if aligned is None:
            continue
        s1, s2, s3 = aligned
        n_valid += 1

        r1 = scores_to_ranks(s1)
        r2 = scores_to_ranks(s2)
        r3 = scores_to_ranks(s3)

        # Kendall's W
        W = kendalls_w(np.stack([r1, r2, r3]))
        Ws.append(W)

        # Pairwise τ（仅点估计，bootstrap 在汇总后做）
        tau12, _, _ = pairwise_kendall_tau(r1, r2, n_boot=0)
        tau13, _, _ = pairwise_kendall_tau(r1, r3, n_boot=0)
        tau23, _, _ = pairwise_kendall_tau(r2, r3, n_boot=0)
        if not np.isnan(tau12):
            taus_12.append(tau12)
        if not np.isnan(tau13):
            taus_13.append(tau13)
        if not np.isnan(tau23):
            taus_23.append(tau23)

        # Top-k agreement
        for k in [3, 5, 10]:
            if len(s1) < k:
                continue
            ag = topk_agreement(s1, s2, s3, k)
            topk_results[k]["12"].append(ag["pairwise_12"])
            topk_results[k]["13"].append(ag["pairwise_13"])
            topk_results[k]["23"].append(ag["pairwise_23"])
            topk_results[k]["3way"].append(ag["three_way"])

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else None

    # Bootstrap CI for each τ list (treating per-problem taus as independent)
    def tau_ci(lst):
        if len(lst) < 5:
            return None
        return bootstrap_ci(np.array(lst))

    return {
        "domain": domain_label,
        "n_problems_used": n_valid,
        "kendalls_W": {
            "values": Ws,
            "mean": safe_mean(Ws),
            "ci95": bootstrap_ci(np.array(Ws)) if Ws else None,
        },
        "pairwise_tau": {
            "m1_m2": {"mean": safe_mean(taus_12), "values": taus_12, "ci95": tau_ci(taus_12)},
            "m1_m3": {"mean": safe_mean(taus_13), "values": taus_13, "ci95": tau_ci(taus_13)},
            "m2_m3": {"mean": safe_mean(taus_23), "values": taus_23, "ci95": tau_ci(taus_23)},
        },
        "topk_agreement": {
            str(k): {key: safe_mean(vals) for key, vals in v.items()}
            for k, v in topk_results.items()
        }
    }


def exp3_cross_domain_W(res_gpqa: dict, res_math: dict) -> dict:
    """Exp 3：比较两域的 Kendall's W（Mann-Whitney U + Cliff's delta）"""
    Ws_gpqa = res_gpqa["kendalls_W"]["values"]
    Ws_math = res_math["kendalls_W"]["values"]

    if not Ws_gpqa or not Ws_math:
        return {"error": "insufficient data"}

    mw_stat, mw_p = mannwhitneyu(Ws_gpqa, Ws_math, alternative="two-sided")
    d = cliffs_delta(np.array(Ws_gpqa), np.array(Ws_math))
    ci_diff = bootstrap_ci(np.array(Ws_gpqa) - np.mean(Ws_math))  # 粗略差异 CI

    return {
        "mann_whitney_u": float(mw_stat),
        "mann_whitney_p": float(mw_p),
        "cliffs_delta": d,
        "effect_size": cliffs_delta_interp(d),
        "mean_W_gpqa": float(np.mean(Ws_gpqa)),
        "mean_W_math": float(np.mean(Ws_math)),
        "ci_diff_approx": ci_diff,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. 打印摘要
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    def fmt(v, digits=4):
        if v is None:
            return "N/A"
        if isinstance(v, (int, float)):
            return f"{v:.{digits}f}"
        return str(v)

    print("\n" + "=" * 70)
    print("Exp 2a — Gini Coefficient")
    print("=" * 70)
    g = results["exp2a"]["gini"]
    print(f"  GPQA  mean={fmt(g['gpqa']['mean'])}  CI={g['gpqa']['ci95']}")
    print(f"  Math  mean={fmt(g['math']['mean'])}  CI={g['math']['ci95']}")
    print(f"  Permutation p={fmt(g['permutation_p'])}  Cliff's δ={fmt(g['cliffs_delta'])} [{g['effect_size']}]")

    print("\n" + "=" * 70)
    print("Exp 2a — Jensen-Shannon Divergence (label distribution)")
    print("=" * 70)
    j = results["exp2a"]["jsd"]
    print(f"  JSD = {fmt(j['value'])}  CI={j['ci95']}")
    print("  Label freq (GPQA vs Math):")
    for lab in j["label_freq_gpqa"]:
        gf = j["label_freq_gpqa"][lab]
        mf = j["label_freq_math"][lab]
        print(f"    {lab:<30s}  gpqa={gf:.4f}  math={mf:.4f}")

    print("\n" + "=" * 70)
    print("Exp 2b — Attention Entropy (receiver head score distribution)")
    print("=" * 70)
    ae = results["exp2b"]
    print(f"  GPQA  mean={fmt(ae['gpqa']['mean'])}  CI={ae['gpqa']['ci95']}")
    print(f"  Math  mean={fmt(ae['math']['mean'])}  CI={ae['math']['ci95']}")
    print(f"  Permutation p={fmt(ae['permutation_p'])}  Cliff's δ={fmt(ae['cliffs_delta'])} [{ae['effect_size']}]")

    print("\n" + "=" * 70)
    print("Exp 2c — Causal Reach Ratio")
    print("=" * 70)
    rr = results["exp2c"]["causal_reach_ratio"]
    print(f"  GPQA  mean={fmt(rr['gpqa']['mean'])}  CI={rr['gpqa']['ci95']}")
    print(f"  Math  mean={fmt(rr['math']['mean'])}  CI={rr['math']['ci95']}")
    print(f"  Permutation p={fmt(rr['permutation_p'])}  Cliff's δ={fmt(rr['cliffs_delta'])} [{rr['effect_size']}]")

    print("\n" + "=" * 70)
    print("Exp 2c — Anchor Location Distribution")
    print("=" * 70)
    al = results["exp2c"]["anchor_location"]
    print(f"  GPQA  n={al['gpqa']['n_anchors']}  mean_pos={fmt(al['gpqa']['mean_pos'])}")
    print(f"  Math  n={al['math']['n_anchors']}  mean_pos={fmt(al['math']['mean_pos'])}")
    print(f"  Mann-Whitney U p={fmt(al['mann_whitney_p'])}  Cliff's δ={fmt(al['cliffs_delta'])} [{al['effect_size']}]")

    print("\n" + "=" * 70)
    print("Exp 3 — Method Agreement (Kendall's W)")
    print("=" * 70)
    for dom in ["gpqa", "math"]:
        wr = results["exp3"][dom]["kendalls_W"]
        print(f"  {dom.upper()}  W_mean={fmt(wr['mean'])}  CI={wr['ci95']}  n_problems={results['exp3'][dom]['n_problems_used']}")

    print("\n" + "=" * 70)
    print("Exp 3 — Pairwise Kendall's τ")
    print("=" * 70)
    for dom in ["gpqa", "math"]:
        pt = results["exp3"][dom]["pairwise_tau"]
        print(f"  {dom.upper()}:")
        for pair, info in pt.items():
            print(f"    {pair}: τ_mean={fmt(info['mean'])}  CI={info['ci95']}")

    print("\n" + "=" * 70)
    print("Exp 3 — Top-k Agreement")
    print("=" * 70)
    for dom in ["gpqa", "math"]:
        print(f"  {dom.upper()}:")
        for k, ag in results["exp3"][dom]["topk_agreement"].items():
            print(f"    k={k}: 12={fmt(ag['12'])} 13={fmt(ag['13'])} 23={fmt(ag['23'])} 3way={fmt(ag['3way'])}")

    print("\n" + "=" * 70)
    print("Exp 3 — Cross-Domain W Comparison")
    print("=" * 70)
    cd = results["exp3"]["cross_domain_W"]
    print(f"  GPQA W_mean={fmt(cd['mean_W_gpqa'])}  Math W_mean={fmt(cd['mean_W_math'])}")
    print(f"  Mann-Whitney U p={fmt(cd['mann_whitney_p'])}  Cliff's δ={fmt(cd['cliffs_delta'])} [{cd['effect_size']}]")

    print("\n")


# ──────────────────────────────────────────────────────────────────────────────
# 8. 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    m1_gpqa = load_method1("gpqa")
    m1_math = load_method1("math")
    m2_gpqa = load_method2("gpqa")
    m2_math = load_method2("math")
    m3_gpqa = load_method3("gpqa")
    m3_math = load_method3("math")
    m3_mat_gpqa = load_method3_matrix("gpqa")
    m3_mat_math = load_method3_matrix("math")

    print(f"  GPQA: {len(m1_gpqa)} problems (m1), "
          f"{len(m2_gpqa)} (m2), {len(m3_gpqa)} (m3)")
    print(f"  Math: {len(m1_math)} problems (m1), "
          f"{len(m2_math)} (m2), {len(m3_math)} (m3)")

    print("Running Exp 2a (Gini + JSD)...")
    res_2a = exp2a_gini_jsd(m1_gpqa, m1_math)

    print("Running Exp 2b (Attention Entropy)...")
    res_2b = exp2b_attention_entropy(m2_gpqa, m2_math, m1_gpqa, m1_math)

    print("Running Exp 2c (Causal Topology)...")
    res_2c = exp2c_causal_topology(m1_gpqa, m1_math, m3_mat_gpqa, m3_mat_math)

    print("Running Exp 3 (Method Agreement) — GPQA...")
    res_3_gpqa = exp3_method_agreement(m1_gpqa, m2_gpqa, m3_gpqa, "gpqa")

    print("Running Exp 3 (Method Agreement) — Math...")
    res_3_math = exp3_method_agreement(m1_math, m2_math, m3_math, "math")

    print("Running Exp 3 cross-domain W comparison...")
    res_3_cross = exp3_cross_domain_W(res_3_gpqa, res_3_math)

    results = {
        "exp2a": res_2a,
        "exp2b": res_2b,
        "exp2c": res_2c,
        "exp3": {
            "gpqa": res_3_gpqa,
            "math": res_3_math,
            "cross_domain_W": res_3_cross,
        }
    }

    # ── 保存结果 ──
    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_domain_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to {out_path}")

    # ── 打印摘要 ──
    print_summary(results)


if __name__ == "__main__":
    main()
