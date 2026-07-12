# Phase 3 Experiment 1: TopK 消融实验 — 归档总结

> **归档日期**: 2026-07-12  
> **实验编号**: Phase 3 / Experiment 1  
> **状态**: 已完成（含分时段复核、口径澄清、缺口拆解）

---

## 1. 实验设置

### 1.1 固定输入与可复现性

| 项目 | 值 |
|---|---|
| 固定预测文件 | `inputs/baseline_pred.pkl` |
| SHA256 | `27f0658bc15ed87392ba87b963b2f639c1c1688d3dcb840f72373bd39b78941d` |
| MD5 | `264d043ee0eded6c1cca46d5e89d223a` |
| 预测行数 / 日期 | 507,193 行；2020-01-02 ~ 2023-12-29 |
| 校验方式 | 四组实验均通过 `--verify-hash`；manifest 见 `inputs/input_hashes.txt` |

四组 TopK 实验（10 / 20 / 50 / 100）**共用同一份** `baseline_pred.pkl`，`pred_hash` 在四份 `summary.json` 中完全一致。

### 1.2 策略与回测参数

| 参数 | 值 |
|---|---|
| 策略 | `TopkDropoutStrategy`（long-only，equal-weight 买入） |
| **TopK** | 10 / 20 / 50 / 100（唯一变量） |
| **n_drop** | **固定 = 2**（所有组相同） |
| risk_degree | 0.95 |
| hold_thresh | 1 |
| method_sell / method_buy | bottom / top |
| forbid_all_trade_at_limit | true |
| 基准 | P10104 |
| 回测窗口 | 2020-01-01 ~ 2023-12-31（test 期） |
| deal_price | close |
| open_cost / close_cost | 0.0001 / 0.0001 |
| Executor | `SimulatorExecutor(time_per_step=day)` |

TopK=20 使用 `configs/baseline.yaml`；其余使用 `configs/topk{10,50,100}.yaml`。除 `topk` 外，其余参数与 baseline 完全一致。

### 1.3 实验方法

- **只重跑 PortAnaRecord 等价路径**（`fill_placeholder` + `qlib.backtest.backtest()` + `risk_analysis()`）
- **不重新训练** LGBModel，不重新生成 pred
- 运行脚本：`scripts/run_portfolio_ablation.py`
- 分时段 / 缺口拆解脚本：`scripts/run_period_analysis.py`、`scripts/decompose_gap.py`

### 1.4 Manifest 修复记录

2026-07-12 21:28 的首次 topk10/50/100 运行因 `configs/topk*.yaml` 未登记入 `input_hashes.txt` 而在 `--verify-hash` 阶段失败（见 `results/topk*.log`）。修复 manifest 后于 21:47 四组全部重跑成功。

---

## 2. 全样本（2020–2023）核心结果

所有 ARR 均为相对 P10104 的**含成本超额年化收益**。

| TopK | ARR (excess, w/ cost) | IR | MDD | Cost Drag (ARR) | 日均换手 | 平均持仓数 |
|---:|---:|---:|---:|---:|---:|---:|
| **10** | **-16.79%** | -0.521 | -91.42% | 0.086% | 3.59% | 11.8 |
| **20** | **-11.13%** | -0.371 | -80.98% | 0.066% | 2.76% | 21.7 |
| **50** | **-14.74%** | -0.555 | -87.28% | 0.012% | 0.50% | 51.8 |
| **100** | **-11.89%** | -0.429 | -72.73% | 0.019% | 0.80% | 101.5 |

**排序（ARR 从高到低）**: TopK=20 > TopK=100 > TopK=50 > TopK=10

### 2.1 U 型现象

全样本呈现非单调的 **U 型（实为 V 型谷底）**：

- **TopK=20 最优**（-11.13%）
- **TopK=50 反而最差**（-14.74%），比 20 差 3.61pp，比 100 差 2.85pp
- **TopK=100 居中**（-11.89%），接近 20 但 MDD 更小（-72.73% vs -80.98%）

U 型**不存在于信号分段 label 空间**（rank 1–20 / 21–50 / 51–100 超额收益单调递减），仅出现在**组合 TopK 层面**。

### 2.2 分时段 IC（信号质量背景）

| 时段 | Mean IC | Mean Rank IC | 交易天数 |
|---|---:|---:|---:|
| 2020Q1（COVID 暴跌） | -0.00672 | -0.00782 | 62 |
| 2020Q2–2021（恢复期） | +0.00259 | +0.00125 | 443 |
| 2022–2023（post-COVID） | +0.00773 | +0.00458 | 501 |

IC 为负仅出现在 2020Q1；但 U 型不能简单归因于 COVID（见第 3 节）。

---

## 3. 分时段结果（2020Q1 / 2020Q2–2021 / 2022–2023）

### 3.1 2020Q1 — COVID 暴跌期（2020-01-01 ~ 2020-03-31，62 天）

| TopK | ARR | IR | MDD | 日均换手 |
|---:|---:|---:|---:|---:|
| 10 | **-144.0%** | -2.31 | -64.2% | 40.1% |
| 20 | -108.7% | -2.69 | -35.7% | 16.8% |
| **50** | **-101.1%** | -2.44 | -34.0% | 7.8% |
| **100** | **-101.5%** | -2.42 | -34.4% | 5.5% |

- **无 U 型**；TopK=50/100 最优（亏损最小），TopK=10 最差
- 高换手（TopK=10 日均 40%）在暴跌期伤害极大

### 3.2 2020Q2–2021 — 暴跌后恢复期（2020-04-01 ~ 2021-12-31，443 天）

| TopK | ARR | IR | MDD | 日均换手 |
|---:|---:|---:|---:|---:|
| 10 | +10.9% | +0.31 | -46.1% | 7.9% |
| **20** | **+14.2%** | +0.46 | -37.2% | 1.5% |
| 50 | +7.5% | +0.26 | -47.7% | 0.76% |
| 100 | +6.0% | +0.22 | -42.4% | 0.51% |

- **单调递减**：20 > 10 > 50 > 100，**无 U 型**
- 该段 443 天，是全样本中**权重最大**的时段
- TopK=50（+7.5%）显著弱于 TopK=20（+14.2%），是全样本 U 型谷底的主要贡献来源

### 3.3 2022–2023 — post-COVID 相对正常期（2022-01-01 ~ 2023-12-31，501 天）

| TopK | ARR | IR | MDD | 日均换手 |
|---:|---:|---:|---:|---:|
| 10 | -20.3% | -0.67 | -85.2% | 23.9% |
| 20 | -11.4% | -0.42 | -68.8% | 12.1% |
| **50** | **-8.2%** | -0.32 | -63.3% | 1.3% |
| 100 | -11.8% | -0.46 | -63.9% | 0.5% |

- **方向反转**：TopK=50 最优（亏损最小），TopK=10 最差
- 无 U 型，但与恢复期结论相反

### 3.4 剔除 2020Q1 后的合并回测（2020-04-01 ~ 2023-12-31，944 天）

| TopK | ARR | IR | MDD | 日均换手 |
|---:|---:|---:|---:|---:|
| 10 | +1.47% | +0.045 | -79.1% | 3.7% |
| 20 | -0.43% | -0.015 | -70.8% | 0.69% |
| **50** | **-5.22%** | -0.191 | -83.2% | 0.35% |
| 100 | -4.42% | -0.171 | -72.9% | 0.24% |

**U 型在剔除 2020Q1 后依然存在**（50 仍比 20 和 100 都差）。

### 3.5 核心结论：Regime-Dependent，无全局最优 TopK

全样本 U 型**不是 COVID 导致的假象**。其机制是三个 market regime 下最优 TopK **方向不一致**，经时段加权后产生的复合效应：

| 时段 | 最优 TopK | 特征 |
|---|---|---|
| 2020Q1 暴跌 | 50 / 100 | 低换手抗暴跌；TopK=10 高换手灾难 |
| 2020Q2–2021 恢复 | **20** | 集中持仓捕获反弹；50 稀释 alpha |
| 2022–2023 正常 | **50** | 适度分散抗波动；10 高换手再受伤 |

**结论：最优 TopK 是 regime-dependent 的，不存在跨周期的全局最优解。**

单时段内均无经典 U 型；全样本 U 型来自跨时段复合，尤其恢复期 TopK=50 对 TopK=20 的显著劣势（+7.5% vs +14.2%）。

---

## 4. Label 分段收益 vs 组合 ARR 的口径澄清

> **重要方法论说明** — 不影响 IC 分析与 TopK 消融结论，无需重跑实验。

### 4.1 最初的口径错误

曾将 **rank 21–50 原始年化收益 +13.33%** 与 **组合 ARR -11.13%** 直接比较。这是错误的：

- 分段收益：股票**绝对收益**（未减 benchmark）
- 组合 ARR：相对 P10104 的**超额收益**（含成本）

两者不可直接对比。

### 4.2 修正后的超额收益分段（全 test 期，1006 天）

| 分段 | 原始年化（不可比） | **超额年化（正确口径）** |
|---|---:|---:|
| rank 1–20 | +18.49% | **-3.05%** |
| rank 21–50 | +13.29% | **-8.24%** |
| rank 51–100 | +12.07% | **-9.47%** |

信号空间单调递减，**无 U 型**。rank 1–20 超额（-3.05%）仍与 TopK=20 组合 ARR（-11.13%）存在约 **8pp 缺口**。

### 4.3 约 8pp 缺口的根因拆解（TopK=20）

拆解层级（年化超额 ARR）：

| 层级 | 定义 | ARR |
|---|---|---:|
| L0 | 当日 score top20 + 当日 label 超额（Task3 理想口径） | -2.43% ~ -3.05% |
| L1 | **T-1 日 score top20** + 当日 label 超额 | **-10.75%** |
| L2 | 真实 n_drop 持仓，等权，label 超额 | -6.19% |
| L8 | 真实组合（无成本） | -10.64% |
| L9 | 真实组合（含成本） | -10.71% |

| 因素 | 贡献 (pp) | 判定 |
|---|---:|---|
| **信号滞后 shift=1**（策略用 T-1 信号，避免 look-ahead） | **~-8.3** | **主因** |
| n_drop 渐进换仓（相对每日追最新排名） | **+4.6** | 噪声平滑，缩小缺口 |
| 权重漂移（非完美等权） | ~-2.0 | 次要 |
| label 公式 vs close-to-close 收益窗口 | ~-1.0 | 次要 |
| **交易成本** | **-0.07** | **可忽略** |

**关键验证**：仅将 score 从当日改为滞后 1 日（L0→L1），ARR 即从 -2.4% 跌至 -10.75%，与组合 -10.71% 几乎重合。

### 4.4 方法论结论

- **8pp 缺口是方法论对齐问题，不是数据或策略 bug**
- IC 分析、TopK 消融、分时段结论**不受此影响**，无需重跑
- Task3 的理想化排名分段收益**不应作为可实现基准写入论文**，应标注为"信号空间参考"；与组合对比须使用滞后信号口径

---

## 5. 意外发现：n_drop 渐进换仓的降噪价值

### 5.1 日度排名剧烈变动

- 真实持仓与**当日 top20 平均重叠率仅 6.3%**
- 与**前一日 top20 重叠率 7.7%**（约 18/20 只持仓不在昨日 top20 内）
- 日度 score 排名波动极大；n_drop=2 每日仅替换约 2 只

案例（2020-01-03 → 2020-01-06）：仅换 2 只（18/20 不变），但与 1/3 当日 top20 重叠仅 8/20。

### 5.2 每日强制再平衡 vs n_drop 渐进换仓

| 策略 | 年化超额 ARR |
|---|---:|
| 每日强制等权再平衡到当日 top20（close-to-close，无成本） | **-16.56%** |
| 现有 n_drop=2 策略（无成本） | **-10.64%** |
| 差距 | **+5.9pp**（n_drop 更优） |

**结论**：信号噪声大、日度排名不稳定的情况下，n_drop 渐进换仓客观起到**平滑/降噪**作用。若每日严格按最新排名全量调仓，表现反而显著恶化。

---

## 6. 待办 / 后续方向

### 6.1 下一步实验

- **Experiment 2：n_drop 消融**（在 TopK=20 固定下，测试 n_drop = 1 / 2 / 5 / 10）

### 6.2 可选深挖

- 验证换仓速度与降噪效果的非线性关系
- 按 regime 分别搜索最优 (TopK, n_drop) 组合
- 将 regime-dependent 发现与 IC 分时段结果联动分析

### 6.3 写作建议

| 内容 | 建议处理 |
|---|---|
| Task3 排名分段超额收益 | 标注为"信号空间参考"，非可实现组合基准 |
| 全样本 U 型 | 解释为三 regime 复合效应，非单一市场现象 |
| Regime-dependent TopK | **可作为章节亮点** |
| 8pp 缺口 | 在方法论章节说明 signal shift=1 对齐问题 |
| n_drop 降噪 | 可作为策略设计层面的意外发现 |

---

## 附录 A：支撑性数据文件清单

### A.1 核心实验输出

| 路径 | 说明 |
|---|---|
| `results/baseline/summary.json` | TopK=20 全样本指标 |
| `results/baseline/effective_config.yaml` | TopK=20 生效配置 |
| `results/topk10/summary.json` | TopK=10 全样本指标 |
| `results/topk10/effective_config.yaml` | TopK=10 生效配置 |
| `results/topk50/summary.json` | TopK=50 全样本指标 |
| `results/topk50/effective_config.yaml` | TopK=50 生效配置 |
| `results/topk100/summary.json` | TopK=100 全样本指标 |
| `results/topk100/effective_config.yaml` | TopK=100 生效配置 |

### A.2 分时段分析（period_analysis）

| 路径 | 说明 |
|---|---|
| `results/period_analysis/task1_ic_by_period.csv` | 三时段 IC / Rank IC |
| `results/period_analysis/task1_ic_by_period.json` | 同上 JSON |
| `results/period_analysis/task2_topk_by_period.csv` | 三时段 × 四 TopK 组合指标 |
| `results/period_analysis/task2_u_shape_analysis.json` | 各时段 U 型检验 |
| `results/period_analysis/task2_combined_excluding_q1.json` | 剔除 Q1 合并回测 |
| `results/period_analysis/task3_excess_segment_summary.csv` | 排名分段超额收益（正确口径） |
| `results/period_analysis/task3_excess_segment_summary.json` | 同上 JSON |
| `results/period_analysis/task3_daily_segment_raw_vs_excess.csv` | 逐日分段原始 vs 超额 |
| `results/period_analysis/task4_manifest_verification.json` | manifest 校验记录 |
| `results/period_analysis/run_metadata.json` | 运行元数据 |

### A.3 缺口拆解（gap_decomposition）

| 路径 | 说明 |
|---|---|
| `results/gap_decomposition/gap_decomposition_summary.json` | 8pp 缺口完整拆解 |
| `results/gap_decomposition/level_metrics.csv` | 各层级年化指标 |
| `results/gap_decomposition/daily_decomposition.csv` | 逐日持仓/重叠/各层级收益 |

### A.4 早期分段分析（topk_segment_analysis）

| 路径 | 说明 |
|---|---|
| `results/topk_segment_analysis/step1_pred_provenance.json` | pred 来源验证 |
| `results/topk_segment_analysis/step2_segment_summary.csv` | 原始口径分段收益 |
| `results/topk_segment_analysis/step2_daily_segment_returns.csv` | 逐日分段原始收益 |
| `results/topk_segment_analysis/step2_excess_segment_summary.csv` | 超额口径分段收益 |
| `results/topk_segment_analysis/step2_excess_daily_segment_returns.csv` | 逐日分段超额收益 |
| `results/topk_segment_analysis/step2_raw_vs_excess_segment_summary.csv` | 原始 vs 超额对比 |
| `results/topk_segment_analysis/step2_worst10_rank21_50_days.csv` | rank 21–50 最差 10 日 |
| `results/topk_segment_analysis/step2_excess_worst10_rank21_50_days.csv` | 超额口径最差 10 日 |
| `results/topk_segment_analysis/step3_*.csv` | 极端 label 深挖 |
| `results/topk_segment_analysis/step4_*.json/csv` | 离群值 trim 敏感性 |

### A.5 输入、配置与脚本

| 路径 | 说明 |
|---|---|
| `inputs/baseline_pred.pkl` | 固定预测（SHA256 见 manifest） |
| `inputs/input_hashes.txt` | SHA256 manifest（5 条目） |
| `configs/baseline.yaml` | TopK=20 配置 |
| `configs/topk10.yaml` | TopK=10 配置 |
| `configs/topk50.yaml` | TopK=50 配置 |
| `configs/topk100.yaml` | TopK=100 配置 |
| `scripts/run_portfolio_ablation.py` | 主运行器 |
| `scripts/run_period_analysis.py` | 分时段 + 分段收益分析 |
| `scripts/decompose_gap.py` | 8pp 缺口拆解 |

### A.6 运行日志

| 路径 | 说明 |
|---|---|
| `results/topk10.log` | 首次失败 + 修复后重跑日志 |
| `results/topk50.log` | 同上 |
| `results/topk100.log` | 同上 |

---

## 附录 B：关键数字速查

```
全样本 ARR:  TopK=20 (-11.13%) > TopK=100 (-11.89%) > TopK=50 (-14.74%) > TopK=10 (-16.79%)
U 型谷底:    TopK=50 比 20 差 3.61pp，比 100 差 2.85pp
Cost Drag:   TopK=20 仅 0.066%（8pp 缺口主因不是成本）
信号滞后:    L0 (-2.4%) → L1 (-10.75%) = -8.3pp
n_drop 降噪: 每日强再平衡 (-16.56%) vs n_drop=2 (-10.64%) = +5.9pp
持仓重叠:    与当日 top20 均值 6.3%
```

---

*本文件为 Phase 3 Experiment 1 的权威归档。后续实验请在本文件基础上追加或引用，避免重复口径错误。*
