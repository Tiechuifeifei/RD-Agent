# Phase 3 组合构建消融 — 综合总结

> **归档日期**: 2026-07-12  
> **状态**: ✅ Phase 3 正式收官（Experiment 1–4 已完成并 commit）  
> **用途**: 论文写作直接引用；整合 TopK / n_drop / hold_thresh / Cost 四条论证链  
> **分实验归档**: [EXPERIMENT1_TOPK_SUMMARY.md](EXPERIMENT1_TOPK_SUMMARY.md) · [experiment2_ndrop/EXPERIMENT2_NDROP_SUMMARY.md](experiment2_ndrop/EXPERIMENT2_NDROP_SUMMARY.md) · [experiment3_holdthresh/EXPERIMENT3_HOLDTHRESH_SUMMARY.md](experiment3_holdthresh/EXPERIMENT3_HOLDTHRESH_SUMMARY.md) · [experiment4_cost/EXPERIMENT4_COST_SUMMARY.md](experiment4_cost/EXPERIMENT4_COST_SUMMARY.md)

---

## Executive Summary

Phase 3 在固定 Alpha20 预测（`baseline_pred.pkl`，test IC = **+0.00458**）的前提下，对 `TopkDropoutStrategy` 的四个组合构建维度做排除法消融。实验 1–3（TopK、n_drop、hold_thresh）一致呈现**非单调、regime-dependent、路径依赖敏感性**；实验 4（交易成本）则呈现**严格单调、近似线性**的对照行为。这一鲜明对比表明：**问题不在于组合执行引擎或成本假设，而在于底层信号强度不足、噪声主导，任何依赖日度选股排名的下游决策都会被放大为不稳定的组合轨迹。** 本系列实验堵住了"是不是组合参数没调好"的质疑，将 RQ1 核心叙事从"组合构建失误"转向"跨市场信号非转移 + 组合层面噪声放大"。

---

## 1. 实验设计总览

### 1.1 为什么选择这四个维度

| 维度 | 实验 | 控制的是什么 | 为何纳入 |
|---|---|---|---|
| **TopK** | Exp1 | 组合宽度 / 信号覆盖范围 | 最直接决定"买多少只股票"；Nasdaq 实验与文献常见调参点 |
| **n_drop** | Exp2 | 日度换仓强度（dropout 数量） | 决定"多快跟随最新排名"；与信号日度波动直接耦合 |
| **hold_thresh** | Exp3 | 最短持有期约束 | 决定"多快允许卖出"；语义清晰，曾假设比 n_drop 更稳定 |
| **Transaction Cost** | Exp4 | 执行层费率（open/close cost） | **对照组**：不改变选股排名，只改变扣费；验证引擎与理论预期 |

四者共同覆盖 `TopkDropoutStrategy` 的核心可调旋钮，且前三者均**影响选股/持仓路径**，第四者**仅影响执行会计**。

### 1.2 方法论：固定 pred.pkl + 只重跑 PortAnaRecord

| 原则 | 做法 | 目的 |
|---|---|---|
| 固定信号 | 全实验共用 `baseline_pred.pkl`（SHA256: `27f0658b...`） | **解耦 IC 与 ARR**：组合差异归因于构造参数，而非模型重训带来的分数变化 |
| 只跑组合层 | `fill_placeholder` + `qlib.backtest()` + `risk_analysis()` | 避免 qrun 重训混淆"信号改进"与"构造改进" |
| 可复现 | `--verify-hash` + `input_hashes.txt` manifest | 每组实验可独立审计 |
| 密集采样 | Exp2/3 吸取教训：直接 11 点网格 | 粗网格可产生错误"最优值"（Exp2：ndrop=10→4） |

**不重新训练**是 Phase 3 的方法论基石：若每组参数重训模型，ARR 差异将混合 IC 漂移与构造效应，无法回答"在**同一信号**下，组合构建是否稳健"。

### 1.3 刻意排除的维度及原因

| 排除项 | 原因 |
|---|---|
| **Long-short** | Qlib US 回测框架对做空撮合支持有限；本仓库 baseline 为 long-only `TopkDropoutStrategy`；引入空头属于**框架扩展**，非当前策略族内的消融 |
| **Strategy 类型替换**（如 WeightStrategy、规则策略） | 更换策略类 = 更换组合构建**范式**，属于 method extension / 新基线对比，而非对 `TopkDropoutStrategy` 的**参数敏感性**分析 |
| **模型重训 / 特征改动** | 属于信号层（IC）实验，非 Phase 3 范围 |

Phase 3 的定位是：**在既定策略类与既定信号下，构造参数是否稳健**——排除法，而非"找最优参数"。

### 1.4 实验规模速查

| 实验 | 网格点数 | 时段 | 总回测数 | 特殊排查 |
|---|---:|---|---:|---|
| Exp1 TopK | 4 | 4 | 16 | 缺口拆解（shift/cost） |
| Exp2 n_drop | 11 | 4 | 44+ | 路径依赖 + ndrop3/4 跳变 |
| Exp3 hold_thresh | 11 | 4 | 44+ | 路径依赖前置（>3pp） |
| Exp4 Cost | 6 | 4 | 24 | 无（关系干净） |

---

## 2. 核心发现横向对比

| 维度 | ARR 是否单调？ | 是否 regime-dependent？ | 路径依赖敏感性？ | 全样本网格最优（pooled） | 分时段最优（示例） |
|---|---|---|---|---|---|
| **TopK** (Exp1) | **否**（U/V 型：20 优、50 谷） | **是** | 未做持仓重叠排查；机制为**选股集合改变** + 跨 regime 复合 | **TopK=20**（-11.13%） | Q1: 50/100；恢复: 20；2022-23: 50 |
| **n_drop** (Exp2) | **否**（锯齿，9 次翻转） | **是** | **是**（EOD 重叠 3–7/20+；nd3↔4 差 7.6pp） | **n_drop=4**（-5.67%） | Q1: 4；恢复: 1；2022-23: 7 |
| **hold_thresh** (Exp3) | **否**（锯齿，5 次翻转） | **是** | **是**（5/5 大跳变配对；重叠 2–9/20+） | **ht=5**（-5.98%） | Q1: 5；恢复: 8；2022-23: 15 |
| **Cost** (Exp4) | **是**（ARR w/ cost 单调↓；Cost Drag 单调↑） | **否** | **否** | **0 bp**（ARR 最高 -11.07%） | 各时段均为 0 bp ARR 最高 |

**读表要点**：

- 前三行"网格最优"均为 **pooled-sample optimum**，**不可外推**为部署默认值
- Exp4 的"最优"仅是"费率越低越好"，无调参意义——它是**敏感性曲线**，不是**寻优问题**
- 只有 Exp4 同时满足：单调 + 无 regime 最优漂移 + 无路径依赖

---

## 3. 核心论证：系统性脆弱性 vs 干净线性对照

### 3.1 两条并列的经验规律

**规律 A — 构造路径参数（TopK / n_drop / hold_thresh）**

1. **非单调**：U 型（TopK）、锯齿（n_drop、hold_thresh）；密集采样后"最优值"可随网格改变（Exp2：5 点→11 点，最优从 ndrop=10 变为 4）
2. **Regime-dependent**：三时段最优方向不一致，全样本曲线 = 多 regime 相反效应的加权复合
3. **路径依赖**（Exp2/3 直接验证）：相邻参数仅差 1 档，数百日后 EOD 持仓重叠可降至个位数/20+，ARR 可差 7–16pp
4. **共同机制**：日度排名信号噪声大（Exp1：相邻日 top20 重叠约 6.3%）→ 微小构造差异 → 轨迹分叉 → 长期 ARR 剧烈分化

**规律 B — 执行会计参数（Transaction Cost）**

1. **严格单调**：Cost Drag 随 bps 线性上升（R²≈1.000，~0.065 pp/bps）；ARR(w/ cost) 严格下降
2. **无 regime 最优漂移**：更高成本在所有时段都更差，无"某个 bps 在 Q1 最优、另一个在 2022 最优"
3. **无路径依赖型锯齿**：无需持仓重叠排查
4. **机制**：成本不改变排名与 dropout 决策；仅扣减收益。换手差异 <0.02pp，来自账户净值反馈的二级效应，非选股改变

### 3.2 鲜明对比的推论

```
若问题出在"回测引擎"或"成本假设"  →  Exp4 应同样出现异常  →  事实：Exp4 最干净
若问题出在"组合参数没调好"        →  应存在可复现的全局最优  →  事实：仅 pooled optimum，regime 间最优漂移
若问题出在"信号弱 + 噪声大"        →  影响路径的参数应脆弱，不影响路径的参数应正常  →  与四实验完全一致
```

**结论（论文可用表述）**：

> Phase 3 表明，在 IC ≈ +0.00458 的弱信号下，组合构建层面对**选股路径**的任何调参都无法得到稳健最优；而对**不影响选股路径**的交易成本，回测响应符合理论预期的线性单调关系。组合层面的亏损（0 成本下全样本 ARR 已为 -11.07%）**不能**通过调节 TopK/n_drop/hold_thresh 系统性修复；其根源在于信号跨市场衰减后，日度排名噪声被路径依赖机制放大。

### 3.3 与 baseline 数字的锚定

| 指标 | 官方 Alpha20 SP500 baseline | Phase 3 启示 |
|---|---|---|
| Test IC | **+0.00458** | 信号弱但为正；非零预测力 |
| Test ARR (excess, w/ cost, 1bp) | **-11.13%** | 组合层大幅亏损 |
| Exp4：0 bp ARR | **-11.07%** | 零成本仍亏 → **非成本驱动** |
| Exp4：1bp Cost Drag | **0.07 pp** | baseline 成本几乎可忽略 |
| Exp1：TopK 20→50 差额 | **~3.6 pp** | 构造选择可造成远大于成本的 ARR 波动 |

IC 为正、ARR 为负的割裂，在 Phase 3 中获得机制性解释：**IC 度量横截面排序的微弱相关性；ARR 度量纵向持仓路径的累积结果——在噪声信号下，后者被路径依赖极度放大。**

---

## 4. 与 RQ1 整体证据链的衔接

### 4.1 RQ1 核心发现回顾

| 证据层 | 内容 |
|---|---|
| 信号层 | Alpha20 Nasdaq-100 IC ≈ **0.023** → SP500 IC ≈ **0.00458**（跨市场非转移/衰减） |
| 组合层 | 同一信号下 SP500 long-only TopK20 ARR ≈ **-11.13%**（IC 与 ARR 背离） |
| 成本层 | RD-Factor 40-loop 平均 cost drag ≈ **0.07 pp**（非主因） |

### 4.2 Phase 3 如何堵住"组合没调好"质疑

审稿人/读者可能质疑：

> "S&P 500 上亏损是因为 TopK、换手、持有期或成本设错了，换个参数就好了。"

Phase 3 的排除法回应：

| 质疑 | Phase 3 回应 |
|---|---|
| "TopK 应该更大/更小" | 4 点网格 + 3 时段：U 型且 regime-dependent；恢复期 20 优、2022-23 又是 50 优——**无全局答案** |
| "换手速度没调好" | 11 点 n_drop：锯齿 + 路径依赖；ndrop=3 vs 4 差 7.6pp 但 2022-23 符号反转——**非稳健可复现** |
| "持有期约束没调好" | 11 点 hold_thresh：同样锯齿 + 路径依赖；ht=4 vs 5 差 15.8pp——**并非更稳定** |
| "成本设太高" | Exp4：0 bp 仍亏 11.07%；1 bp 仅 0.07 pp drag——**成本不是主因** |
| "回测引擎有问题" | Exp4 线性单调符合理论——**引擎在正确参数上行为正常** |

**衔接句（论文可用）**：

> Phase 3 并不试图为 Alpha20 在 S&P 500 上寻找"更好的组合参数"，而是系统性地证明：**在固定弱信号（IC = 0.00458）下，不存在稳健的构造参数最优解。** 这与 Nasdaq-100 → S&P 500 的信号衰减叙事一致：问题首先是**信号跨市场不可转移**，其次才是组合层在噪声信号上的**不可避免的路径放大**。

### 4.3 证据链全景（Phase 3 在其中的位置）

```
Nasdaq IC 0.023  →  SP500 IC 0.00458          [信号非转移]
        ↓
SP500 IC > 0 但 ARR << 0                       [IC–ARR 背离，官方 baseline]
        ↓
Cost drag ≈ 0.07 pp（40-loop + Exp4）          [排除成本主因]
Shift-1 解释 ~8pp 缺口（Exp1 拆解）            [排除即时执行/标签口径主因]
TopK / n_drop / hold_thresh 均脆弱（Exp1–3）   [排除"调参可救"]
Cost 维度干净单调（Exp4）                      [排除引擎异常]
        ↓
结论：弱信号 + 路径依赖 → 组合层系统性亏损     [RQ1 机制闭合]
```

---

## 5. 写作建议

### 5.1 推荐定位（用语）

| 推荐 | 避免 |
|---|---|
| "排除法验证（elimination study）" | "参数优化实验" |
| "系统性脆弱性（systematic construction fragility）" | "我们发现了最优 TopK/n_drop" |
| "Pooled-sample optimum（样本内网格峰值）" | "全局最优参数" |
| "路径依赖敏感性（path-dependency sensitivity）" | "参数敏感"（不解释机制） |
| "Regime-dependent，无 universal optimum" | "在测试集上表现最好的设置" |
| Exp4："sanity-check / contrast experiment" | 把 Cost 实验说成"第四组寻优" |

### 5.2 建议的论文段落结构

1. **动机段**：IC 正、ARR 负 → 需排除组合构建失误假说  
2. **方法段**：固定 pred、PortAna-only、四维度消融、为何排除 long-short / 换策略  
3. **结果段 A**：Exp1–3 横向对比表（非单调 + regime + 路径依赖）  
4. **结果段 B**：Exp4 对照（线性 Cost Drag；0 bp 仍亏）  
5. **机制段**：信号噪声 → 排名日度剧变 → 路径分叉 → ARR 分化  
6. **结论段**：不能通过调参"救" SP500 组合；与跨市场 IC 衰减一致  

### 5.3 可直接引用的结论句

**英文草稿**：

> We conduct a four-way portfolio-construction ablation on S&P 500 (2020–2023) holding model predictions fixed (IC = 0.00458). Parameters that alter the stock-selection path (TopK, n_drop, hold_thresh) exhibit non-monotonic ARR, regime-dependent optima, and path-dependency sensitivity. In contrast, transaction-cost scaling yields strictly monotone, near-linear cost drag (R² ≈ 1.0), with identical ranking logic across cost levels. At zero transaction cost, annualized excess return remains −11.07%, indicating that underperformance is not driven by fee assumptions or coarse parameter tuning. These results support the view that Alpha20's U.S. failure reflects weak, non-transferable signal amplified by noisy daily rankings—not a fixable portfolio-configuration error.

**中文对应**：

> 我们在 S&P 500（2020–2023）上进行了四维度组合构建排除法实验，固定模型预测（IC = 0.00458）。改变选股路径的参数（TopK、n_drop、hold_thresh）均呈现非单调 ARR、regime-dependent 最优和路径依赖敏感性；而交易成本缩放产生严格单调、近似线性的 cost drag（R²≈1.0），且不改变排名逻辑。零成本下年化超额收益仍为 -11.07%，表明亏损并非费率假设或粗调参所致。结果支持：Alpha20 在美国市场的失效反映的是**弱信号、不可转移、被日度排名噪声放大**——而非可修复的组合配置错误。

### 5.4 Cost 实验的叙事角色（勿省略）

Exp4 不是"凑数第四组"，而是**证伪对照**：

- 若四组都锯齿 → 读者可质疑回测框架  
- 若三组锯齿、一组线性 → 脆弱性与**路径**绑定，框架可信  
- 论文中建议用 **1–2 句** 点明："The cost sweep serves as a positive control: the backtest engine behaves as expected when portfolio rankings are held fixed."

### 5.5 数字引用规范

- 报告 TopK/n_drop/hold_thresh 最优时，**必须**注明"pooled-sample grid optimum on 2020–2023"并给出网格密度（4 点或 11 点）  
- 同时报告**至少一个**分时段最优反例（如 n_drop：全样本 4 vs 恢复 1）  
- IC 引用官方值 **0.00458**；ARR 引用 baseline **-11.13%**（1 bp）与 Exp4 零成本 **-11.07%**  

---

## 附录：分实验归档索引

| 实验 | 总结文档 | 关键脚本 |
|---|---|---|
| Exp1 TopK | [EXPERIMENT1_TOPK_SUMMARY.md](EXPERIMENT1_TOPK_SUMMARY.md) | `run_portfolio_ablation.py`, `run_period_analysis.py` |
| Exp2 n_drop | [experiment2_ndrop/EXPERIMENT2_NDROP_SUMMARY.md](experiment2_ndrop/EXPERIMENT2_NDROP_SUMMARY.md) | `run_experiment2_ndrop.py`, `run_experiment2_densify.py`, `investigate_ndrop34_jump.py` |
| Exp3 hold_thresh | [experiment3_holdthresh/EXPERIMENT3_HOLDTHRESH_SUMMARY.md](experiment3_holdthresh/EXPERIMENT3_HOLDTHRESH_SUMMARY.md) | `run_experiment3_holdthresh.py` |
| Exp4 Cost | [experiment4_cost/EXPERIMENT4_COST_SUMMARY.md](experiment4_cost/EXPERIMENT4_COST_SUMMARY.md) | `run_experiment4_cost.py` |
| 方法论决策 | `wrds_project/docs/decisions.md` § Phase 3 | — |

### 关键数字速查（Phase 3 全集）

```
信号:     IC = +0.00458 (Alpha20 SP500 test)
Baseline: ARR = -11.13% (topk=20, n_drop=2, ht=1, 1bp)

Exp1 全样本最优:  TopK=20 (-11.13%)
Exp2 全样本最优:  n_drop=4 (-5.67%)   [11点]
Exp3 全样本最优:  ht=5 (-5.98%)       [11点]
Exp4 零成本 ARR:  -11.07%             [Cost Drag 线性, R²≈1]

路径依赖: Exp2/3 确认；Exp1 选股集合机制；Exp4 无
Regime:   Exp1-3 是；Exp4 否
```

---

*Phase 3 组合构建消融正式收官。本文件为四实验的权威综合归档。*
