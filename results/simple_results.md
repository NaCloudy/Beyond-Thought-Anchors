**Results summary (for report reference):**

| Metric                 | Finding                                                      |
| ---------------------- | ------------------------------------------------------------ |
| Gini coefficient       | GPQA = 0.864, Math = 0.848, no significant difference (p = 0.48, δ = 0.11, negligible). Both domains exhibit strong anchor concentration. |
| JSD label distribution | JSD = 0.095 [0.084, 0.108]. Largest gaps: plan_generation (Math 11.6% vs GPQA 1.1%), self_checking (Math 8.4% vs GPQA 0.6%), uncertainty_management (GPQA 29.0% vs Math 11.8%). |
| Anchor position        | GPQA anchors appear significantly later (mean = 0.36) than Math (mean = 0.28); Mann-Whitney p = 0.0009, Cliff's δ = 0.18 (small). This is the strongest cross-domain difference. |
| Kendall's W            | Moderate three-way concordance in both domains (MATH W = 0.346, GPQA W = 0.332); cross-domain difference not significant (p = 0.84, δ = 0.04, negligible). |
| Pairwise τ             | M1–M2 (black-box vs receiver head) is the most consistent pair: τ = +0.102 on MATH, +0.141 on GPQA (CIs exclude zero). M1–M3 (black-box vs causal masking): weakly positive on MATH (τ = +0.046, CI [0.007, 0.082]) but non-significant on GPQA (τ = -0.019, CI crosses zero). M2–M3: uncorrelated in both domains. |

