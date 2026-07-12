# Phase 3 Experiment 3: hold_thresh（持仓约束）消融 — 归档总结

> **归档日期**: 2026-07-12  
> **实验编号**: Phase 3 / Experiment 3  
> **状态**: ✅ 已完成并归档（全样本 11 点、三时段同步、路径依赖前置排查）  
> **前置实验**: [EXPERIMENT1_TOPK_SUMMARY.md](../EXPERIMENT1_TOPK_SUMMARY.md) · [EXPERIMENT2_NDROP_SUMMARY.md](../experiment2_ndrop/EXPERIMENT2_NDROP_SUMMARY.md)

---

## Executive Summary

在固定 `baseline_pred.pkl`（SHA256: `27f0658b...`）、`topk=20`、`n_drop=2` 的条件下，对 hold_thresh 进行 **11 点密集采样**（1,2,3,4,5,6,8,10,15,20,30），44 组回测（全样本 + 三时段）一次性完成。全样本 ARR 仍呈锯齿状（5 次翻转），网格最优为 **hold_thresh=5（-5.98%）**，最深谷为 **hold_thresh=4（-21.80%）**——**hold_thresh 并未表现出比 TopK/n_drop 更稳定的单调关系**。换手率和成本拖累在粗粒度上随 hold_thresh 增大而下降，但相邻点存在 4 处局部违反单调性。5 对相邻参数（>3pp 跳变）的路径依赖排查均显示 EOD 持仓重叠仅 2–9/20+ 只，**同样存在路径依赖敏感性**。最优 hold_thresh 强烈 regime-dependent（Q1=5、恢复=8、2022-23=15、全样本=5），不支持跨 regime 全局最优。

---

## 1. 实验设置

### 1.1 固定输入与可复现性

| 项目 | 值 |
|---|---|
| 固定预测文件 | `inputs/baseline_pred.pkl` |
| SHA256 | `27f0658bc15ed87392ba87b963b2f639c1c1688d3dcb840f72373bd39b78941d` |
| 校验方式 | 跑前批量 `--verify-hash`（12 配置 + pred）；见 `manifest_verification.json` |
| topk | **固定 = 20** |
| n_drop | **固定 = 2**（baseline） |

其余参数（risk_degree=0.95、deal_price=close、成本费率、回测窗口 2020–2023）与 baseline 完全一致。

### 1.2 变量与对照

hold_thresh = **1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30**

| hold_thresh | 配置文件 | 说明 |
|---:|---|---|
| **1** | `configs/baseline.yaml` | **baseline**（默认，无实质持有约束） |
| 2 | `configs/holdthresh2.yaml` | 至少持有 2 日 |
| 3 | `configs/holdthresh3.yaml` | |
| 4 | `configs/holdthresh4.yaml` | 全样本网格最差（-21.80%） |
| 5 | `configs/holdthresh5.yaml` | **全样本网格最优（-5.98%）** |
| 6 | `configs/holdthresh6.yaml` | |
| 8 | `configs/holdthresh8.yaml` | 恢复期网格最优 |
| 10 | `configs/holdthresh10.yaml` | |
| 15 | `configs/holdthresh15.yaml` | 2022-23 网格最优 |
| 20 | `configs/holdthresh20.yaml` | |
| 30 | `configs/holdthresh30.yaml` | 极端长持对照 |

### 1.3 实验方法

- **固定 `baseline_pred.pkl`，只重跑 PortAnaRecord**，不重新训练模型
- 运行脚本：`scripts/run_experiment3_holdthresh.py`（44 组一次性：全样本 + 三时段 × 11 点）
- 单组运行器：`scripts/run_portfolio_ablation.py --verify-hash`
- hold_thresh=1 复用 baseline 配置；各时段**独立初始化持仓**
- 路径依赖排查：**前置执行**（阈值相邻 |ARR 差| > 3pp），不等观察锯齿后再排查

### 1.4 Manifest 登记

`inputs/input_hashes.txt` 新增 10 条 holdthresh 配置 SHA256；批量校验记录见 `experiment3_holdthresh/manifest_verification.json`。

---

## 2. 任务1：全样本（2020–2023）11 点结果

| hold_thresh | ARR (w/ cost) | ARR (w/o cost) | Cost Drag | IR | MDD | 日均换手 | 平均持仓 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (baseline) | -11.13% | -11.07% | 0.066% | -0.371 | -81.0% | 2.76% | 21.7 |
| 2 | -10.25% | -10.19% | 0.053% | -0.329 | -65.7% | 2.22% | 21.8 |
| 3 | -9.44% | -9.39% | 0.053% | -0.300 | -84.7% | 2.22% | 21.7 |
| **4** | **-21.80%** | -21.74% | 0.063% | -0.699 | **-108.7%** | 2.63% | 21.7 |
| **5** | **-5.98%** | -5.95% | 0.026% | -0.218 | -61.1% | **1.08%** | 21.9 |
| 6 | -14.81% | -14.78% | 0.033% | -0.488 | -82.2% | 1.37% | 21.8 |
| 8 | -15.10% | -15.08% | 0.022% | -0.475 | -83.9% | 0.91% | 21.9 |
| 10 | -13.12% | -13.08% | 0.046% | -0.409 | -81.3% | 1.95% | 21.8 |
| 15 | -9.96% | -9.92% | 0.033% | -0.314 | -74.2% | 1.40% | 21.8 |
| 20 | -7.63% | -7.61% | 0.018% | -0.228 | -70.4% | 0.76% | 21.9 |
| 30 | -14.60% | -14.58% | 0.014% | -0.470 | -78.6% | **0.57%** | 21.9 |

**平均持仓数 21.7–21.9**，topk=20 恒定，无异常。

### 2.1 ARR 折线形状

```
hold_thresh:  1    2    3    4    5    6    8   10   15   20   30
ARR(%):    -11.1 -10.2 -9.4 -21.8 -6.0 -14.8 -15.1 -13.1 -10.0 -7.6 -14.6
走势:           ↑    ↑    ↓    ↑    ↓    ↓    ↑    ↑    ↑    ↓
方向序列: ↑↑↓↑↓↓↑↑↑↓  （10 段中 5 次翻转）
```

**判定：锯齿状，非单调。** 与 n_drop 消融类似，存在深谷（ht=4）和局部峰（ht=5、ht=20）。

折线图：`arr_vs_holdthresh_full_sample.png`、`arr_vs_holdthresh_four_panel.png`

---

## 3. 任务2：分时段结果（11 点同步）

### 3.1 各时段最优 hold_thresh

| 时段 | 天数 | 最优 ht | 最优 ARR | 最差 ht | 最差 ARR | 翻转次数 |
|---|---:|---:|---:|---:|---:|---:|
| 全样本 | 1006 | **5** | -5.98% | **4** | -21.80% | 5 |
| 2020Q1 COVID | 62 | **5** | -50.32% | 30 | -151.49% | 6 |
| 2020Q2-2021 恢复 | 443 | **8** | +21.65% | 4 | +0.86% | 8 |
| 2022-2023 | 501 | **15** | +2.18% | 30 | -21.08% | 7 |

**结论：强烈 regime-dependent。** 全样本与 Q1 同为 ht=5，但恢复期最优 ht=8，2022-23 最优 ht=15。

### 3.2 分时段 ARR 快照（selected）

| ht | 全样本 | 2020Q1 | 2020Q2-21 | 2022-23 |
|---:|---:|---:|---:|---:|
| 1 | -11.13% | -108.72% | +14.19% | -11.45% |
| 4 | -21.80% | -132.36% | +0.86% | -4.13% |
| 5 | -5.98% | -50.32% | +10.83% | -14.43% |
| 8 | -15.10% | -115.59% | **+21.65%** | -3.12% |
| 15 | -9.96% | -142.38% | +12.85% | **+2.18%** |
| 30 | -14.60% | -151.49% | +5.35% | -21.08% |

---

## 4. 任务3：路径依赖前置排查

对全样本相邻点 |ARR 差| > 3pp 的 **5 对**全部执行排查：

| 配对 | ARR 差 (pp) | top20 日均重叠 | 全样本中位重叠 | 路径依赖？ |
|---|---:|---:|---:|---|
| ht3 → ht4 | -12.36 | 8.5 | 8 | ✅ 是 |
| ht4 → ht5 | +15.82 | 6.0 | 6 | ✅ 是 |
| ht5 → ht6 | -8.84 | 5.0 | 4 | ✅ 是 |
| ht10 → ht15 | +3.17 | 3.0 | 2 | ✅ 是 |
| ht20 → ht30 | -6.97 | 5.2 | 5 | ✅ 是 |

**共同特征**（与 Exp2 n_drop 发现一致）：
- 高差额日 EOD 持仓重叠仅 **2–9 / 20+** 只
- 相邻 hold_thresh 差 1 日约束 → 长期路径分叉 → 组合轨迹大幅分化
- **5/5 配对均判定为路径依赖敏感性**

输出目录：`path_dependency_checks/ht{lo}_vs_ht{hi}/`

---

## 5. 任务4：核心问题回答

### Q1：hold_thresh 增大是否降低换手和成本拖累？

| 维度 | 全样本 11 点 | 判定 |
|---|---|---|
| 日均换手 | ht=1: 2.76% → ht=30: 0.57%（降幅 79%） | **粗粒度成立** |
| 相邻单调性 | 4 处违反（2→3, 3→4, 5→6, 8→10 换手上升） | **非严格单调** |
| Cost Drag | ht=1: 0.066% → ht=30: 0.014%（降幅 79%） | **粗粒度成立** |
| 相邻单调性 | 同换手，4 处违反 | **非严格单调** |

**结论**：直觉方向正确（长期持有 → 更低换手/成本），但 11 点密集采样揭示**局部非单调**，原因仍是路径依赖导致不同 ht 走不同交易轨迹。

2022-23 子时段换手单调性更好（ht=1 12.1% → ht=30 1.9%，基本单调下降）。

### Q2：hold_thresh 对 ARR 是否更稳定？

| 对比维度 | TopK (Exp1) | n_drop (Exp2) | hold_thresh (Exp3) |
|---|---|---|---|
| 全样本折线 | U 型 | 锯齿（9 翻转） | 锯齿（5 翻转） |
| regime-dependent | 是 | 是 | **是** |
| 路径依赖 | — | 确认 | **确认** |
| 相邻跳变 | 大 | 7.6pp (nd3→4) | **15.8pp (ht4→5)** |

**结论：hold_thresh 同样表现出锯齿状和路径依赖，并非更"稳定"的参数。** 翻转次数略少于 n_drop（5 vs 9），但相邻跳变更大（15.8pp vs 7.6pp）。

### Q3：极端值 ht=1 vs ht=30

| 时段 | ht=1（无约束） | ht=30（近强制长持） | 更优 |
|---|---:|---:|---|
| 全样本 | -11.13% | -14.60% | **ht=1** |
| 2020Q1 | -108.72% | -151.49% | **ht=1** |
| 2020Q2-21 | +14.19% | +5.35% | **ht=1** |
| 2022-23 | -11.45% | -21.08% | **ht=1** |

**结论**：极端长持（ht=30）在**所有时段均不优于**无约束（ht=1）。强制长期持有并未带来稳健改善；在 COVID 和 2022-23 反而显著更差。

---

## 6. 与 Experiment 1/2 的交叉验证

| 共同发现 | Exp1 (TopK) | Exp2 (n_drop) | Exp3 (hold_thresh) |
|---|---|---|---|
| regime-dependent 最优 | Q1=50, 恢复=10, 2022-23=20 | Q1=4, 恢复=1, 2022-23=7 | Q1=5, 恢复=8, 2022-23=15 |
| 无跨 regime 全局最优 | 确认 | 确认 | 确认 |
| ARR 折线非单调 | U 型 | 锯齿（9 翻转） | 锯齿（5 翻转） |
| 路径依赖敏感性 | — | EOD 重叠 3–7/20+ | EOD 重叠 2–9/20+ |
| 交易成本非 ARR 主因 | 确认 | 确认 | 确认（Cost Drag 最大 0.066%） |

---

## 7. 最终结论（整合 §1–§6）

### 7.1 实验设计与可复现性

- 吸取 Exp2 教训：**直接 11 点密集采样**，无"先粗测后加密"两阶段
- 44 组回测（4 时段 × 11 点）+ 5 对路径依赖排查，一次跑完
- 固定 pred（SHA256 `27f0658b...`），topk=20，n_drop=2

### 7.2 全样本 ARR 形态

- 11 点 ARR 呈**锯齿状**（↑↑↓↑↓↓↑↑↑↓，5 次翻转），非单调
- 网格最优：**hold_thresh=5（-5.98%）**；网格最差：**hold_thresh=4（-21.80%）**
- 相邻跳变最大：**ht4→5（+15.8pp）**，超过 Exp2 nd3→4（7.6pp）

### 7.3 换手/成本（Q1）

- **粗粒度成立**：ht 1→30，换手 2.76%→0.57%，Cost Drag 0.066%→0.014%
- **细粒度非单调**：相邻点 4 处违反（路径依赖导致不同交易轨迹）

### 7.4 路径依赖（Q2/Q3 机制）

- 5/5 大跳变配对均显示低 EOD 持仓重叠（2–9/20+）
- 机制与 Exp2 n_drop **同构**：微小约束差异 → 长期轨迹分叉 → ARR 剧烈分化
- hold_thresh **未**表现出比 TopK/n_drop 更稳定的 ARR 响应

### 7.5 极端值与 regime-dependent

- **ht=1 vs ht=30**：全样本及三时段均为 ht=1 更优；强制长持无稳健收益
- 最优 ht：全样本/Q1=5，恢复=8，2022-23=15——**无全局最优**

### 7.6 论文级结论定位

> hold_thresh 消融完成 Phase 3 三角验证（TopK / n_drop / hold_thresh）。三个组合构造维度均呈现 regime-dependent 最优、锯齿状 ARR 响应和路径依赖敏感性。**组合层面不存在稳健可靠的参数最优规律**；收益差异主要由信号噪声经路径依赖放大，而非可发现的全局构造法则。

---

## 8. 实验演进

| 阶段 | 内容 | 状态 |
|---|---|---|
| 设计 | 吸取 Exp2 教训，直接 11 点密集采样 | ✅ |
| 44 组回测 | 全样本 + 三时段同步 | ✅ |
| 路径依赖前置排查 | 5 对 >3pp 配对全部排查 | ✅ |
| 正式归档 | 本文档 + `docs/decisions.md` | ✅ |

---

## 附录 A：支撑性数据文件清单

### A.1 本实验输出（experiment3_holdthresh/）

| 路径 | 说明 |
|---|---|
| `task_all_runs.csv` / `.json` | 44 组全指标 |
| `curve_analysis.json` | 各时段折线分析 |
| `adjacent_gaps_full_sample.json` | 相邻点 ARR 差距 |
| `key_findings.json` | 任务4 综合结论 |
| `manifest_verification.json` | 批量 hash 校验 |
| `arr_vs_holdthresh_full_sample.png` | 全样本折线图 |
| `arr_vs_holdthresh_four_panel.png` | 四时段对比 |
| `turnover_cost_vs_holdthresh.png` | 换手/成本 vs ht |
| `path_dependency_checks/ht3_vs_ht4/` | ht3 vs ht4 路径依赖排查 |
| `path_dependency_checks/ht4_vs_ht5/` | ht4 vs ht5 路径依赖排查 |
| `path_dependency_checks/ht5_vs_ht6/` | ht5 vs ht6 路径依赖排查 |
| `path_dependency_checks/ht10_vs_ht15/` | ht10 vs ht15 路径依赖排查 |
| `path_dependency_checks/ht20_vs_ht30/` | ht20 vs ht30 路径依赖排查 |
| `run.log` | 完整运行日志 |

### A.2 配置与脚本

| 路径 | 说明 |
|---|---|
| `configs/holdthresh2.yaml` … `holdthresh30.yaml` | hold_thresh 配置（10 个） |
| `configs/baseline.yaml` | hold_thresh=1 (baseline) |
| `inputs/input_hashes.txt` | 更新后的 manifest（25 条目） |
| `inputs/baseline_pred.pkl` | 固定预测（gitignore，SHA256 见 manifest） |
| `scripts/run_experiment3_holdthresh.py` | 批量运行 + 分析 + 路径依赖脚本 |
| `scripts/run_portfolio_ablation.py` | 单组 PortAna 运行器 |

### A.3 前置实验交叉引用

| 路径 | 说明 |
|---|---|
| `results/EXPERIMENT1_TOPK_SUMMARY.md` | TopK 消融归档 |
| `results/experiment2_ndrop/EXPERIMENT2_NDROP_SUMMARY.md` | n_drop 消融归档 |
| `results/baseline/summary.json` | hold_thresh=1 原始输出 |

## 附录 B：关键数字速查

```
全样本最优:   hold_thresh=5  (-5.98%)
全样本最差:   hold_thresh=4  (-21.80%)
全样本折线:   ↑↑↓↑↓↓↑↑↑↓  (5 flips)
最大相邻跳变: ht4→5 (+15.8pp)
换手(粗趋势): ht1 2.76% → ht30 0.57%
分时段最优:   Q1=5 / 恢复=8 / 2022-23=15 / 全样本=5
路径依赖:     5/5 配对确认 (overlap 2-9/20+)
```

---

*本文件为 Phase 3 Experiment 3 的权威归档（最终版）。Experiment 1 见 `EXPERIMENT1_TOPK_SUMMARY.md`；Experiment 2 见 `experiment2_ndrop/EXPERIMENT2_NDROP_SUMMARY.md`。方法论决策见 `wrds_project/docs/decisions.md`。*
