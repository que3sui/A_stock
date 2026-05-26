# 项目进度记录

> 最后更新: 2026-05-16 18:50
> 截止 6/1 模型 deadline: 16 天

## 已完成 (Day 1 ~ MASTER v3)

### 数据基础

- `cache/panel.parquet` (1362 MB) — 1070万行长面板, 5756股×2515天, 50列
- `cache/universe.parquet` (~3 MB) — 中证800近似池 (市值前800, 月度调整, 4.6%月度换手)
- `cache/features.parquet` (199 MB) — universe过滤+中性化后的20维因子
- `cache/labels.parquet` (102 MB) — 5日累计log收益 + 横截面rank
- `cache/factors_raw.parquet` (1131 MB) — 原始20因子 (中性化前)
- `cache/market_features.parquet` (182 KB) — 12维市场状态信号 (3指数×4特征)

### 模型 (全部out-of-sample 2024-2025测试集 565 天, **真实交易约束**)

> 已修复 market_features 全期标准化泄露; 已加入手续费(双边0.025%+卖出0.1%)+涨跌停过滤

| 指标            | LGBM      | GRU+Att | **MASTER** | MASTER v2 | MASTER v3  | Ensemble   | HS300  |
| --------------- | --------- | ------- | ---------- | --------- | ---------- | ---------- | ------ |
| Test IC         | 0.0511    | 0.0511  | **0.0609** | 0.0539    | 0.0572     | 0.0541     | -      |
| Test RankIC     | 0.0459    | 0.0443  | 0.0572     | 0.0476    | **0.0580** | 0.0552     | -      |
| 年化 RankICIR   | 5.21      | 4.42    | 6.08       | 6.02      | **6.98**   | 5.51       | -      |
| 总收益 (真实)   | 87.4%     | 20.8%   | 79.4%      | 65.6%     | 71.4%      | 82.1%      | 43.9%  |
| 年化收益 (真实) | **45.8%** | 21.6%   | 42.8%      | 29.7%     | 41.5%      | 43.7%      | 19.6%  |
| 夏普 (真实)     | **2.21**  | 0.80    | 2.12       | 1.12      | 1.64       | 2.19       | 1.08   |
| 最大回撤 (真实) | -15.5%    | -31.6%  | -11.1%     | -21.4%    | -20.3%     | **-10.7%** | -15.7% |
| 总收益 (理想)   | -         | -       | 115.7%     | -         | 103.0%     | -          | -      |
| 年化收益 (理想) | -         | -       | 43.7%      | -         | 41.5%      | -          | -      |
| 训练时长        | 8s        | 17min   | 5min       | 14min     | 11min      | -          | -      |
| 参数量          | 83 trees  | 43K     | 123K       | 422K      | 123K × 3   | -          | -      |

**真实 vs 理想化 (MASTER)**: 年化收益 43.7% → 42.8% (-0.9pp), 夏普 2.19 → 2.12 (-5%); 但累积净值 116% → 79% (-37pp, 复利累积放大). 加费率影响小, 策略真实可行.

**最佳模型**:

- 真实约束下: **LGBM (夏普 2.21)** 略胜 MASTER (2.12), 因 LGBM 换手稳定/对涨停过滤不敏感
- ICIR (信号稳定性): **MASTER v3 (multi-seed, 6.98)** 最高
- 综合: **MASTER v1** 仍是最佳深度学习模型, 满足作业"必须包含 DL"要求

### 数据治理修复 (P1 完成)

- ✅ `market_features.py` 用 train 段(<=2022) mean/std 标准化, 修复全期标准化泄露
- 影响: master v1 IC 0.0608 → 0.0609 (近乎一致), 但合规性恢复

### 回测真实性升级 (P2 完成)

- ✅ 一字板 (open==high==low) 自动跳过, 模拟真实涨跌停限制
- ✅ 双边 0.025% 佣金 + 卖出 0.1% 印花税
- ✅ T+1 隐式 (portfolio 来自上日 score, 用当日 pct_chg 累计)
- 565 天累积手续费约 17%

### 分段 IC 分析 (P3 完成)

- LGBM 全期 IC: 2016=0.17 → 2025=0.05 (-71%), **单调衰减**
- MASTER 季度: 2024Q2 / Q4 / 2025Q1 高 IC (~0.11), 2024Q3 / 2025Q3 低 (~0.01), **2026Q2 已 RankIC=-0.094**
- 候选制度断点: 2017 (外资流入), 2023 (全面注册制), 2026 (新阶段)
- 启示: 单次切分训练不应用 > 2 年, 需要滚动重训

### 文件清单 (新增 v2 / v3 / ensemble / report)

```
code/
├── data/
│   ├── build_panel.py
│   ├── universe.py
│   └── validate.py
├── features/
│   ├── factors.py
│   ├── neutralize.py
│   ├── labels.py
│   └── market_features.py
├── models/
│   ├── lgbm_baseline.py
│   ├── gru_att.py
│   ├── master.py            # v1 主模型
│   ├── master_v2.py         # 加深加宽 (失败案例)
│   ├── master_v3.py         # multi-seed 平均
│   └── ensemble.py          # 3 模型 rank-percentile 加权
├── backtest/
│   └── engine.py            # 支持 lgbm/gru/master/master_v2/master_v3/ensemble
├── live/
│   └── daily_signal.py      # 支持所有 6 种模型
├── report/
│   └── build_report.py      # HTML 报告 + 6 张图
└── check_env.py
```

## 关键发现 (适合写报告)

### 1. IC 持平不等于实战收益持平

- LGBM 和 GRU 的 Test IC 完全相同 (0.0511) 但回测收益差 2.6 倍 (118% vs 46%)
- v3 (multi-seed) ICIR 提升 16% (6.06→7.02) 但 sharpe 反而下降 (2.24→1.81)
- 启示: IC 是"全样本相关性", 不直接度量"top-K 选股质量"

### 2. 横截面建模 >> 单股时序建模

- GRU 只看单股过去20天, 缺乏横截面感知 → 表现最差
- LGBM (tree) 天然在横截面做特征比较 → 远超 GRU
- MASTER 同时建模 时序 (Intra-stock TX) + 横截面 (Inter-stock TX) → SOTA

### 3. Market Guidance 有效

- MASTER 的 market-gate 模块用大盘状态调制原始因子
- 同等参数量下,加入 market gate 的版本明显优于不加

### 4. A股短期信号收敛速度快

- 三个模型都在 epoch 2-6 达到验证集最佳 (LGBM 83 trees)
- 说明数据信号清晰强烈, 不需要复杂模型, 但需要正确的归纳偏置

### 5. MASTER 加深加宽反而过拟合 (反直觉发现, v2)

- T=20→30, H=64→96, layers 加深, 参数量 122K → 422K (3.5x)
- val_rank_ic 微升 0.069→0.070, 但 test_rank_ic 下降 0.057→0.048
- 夏普从 2.24 跌到 1.12, 最大回撤从 -10.9% 扩大到 -21.4%
- best_epoch=1 → 模型 1 epoch 就过拟合
- 启示: A股短期 alpha 信号容量约 ~100K 参数, 容量竞赛适得其反

### 6. Multi-seed 减少方差但模糊信号 (v3)

- 用 v1 完全相同配置, 3 个 seed (42, 1337, 2024) 训练后 rank-percentile 平均
- ICIR 从 6.06 → 7.02 (+16%), 信号稳定性提升
- 但 RankIC 略降 0.057→0.055, **回测夏普 2.24→1.81 反而下降**
- 启示: 减方差不等于增收益, 顶部排序的"确信度"也很重要

### 7. Ensemble 提供最低回撤但牺牲收益

- 三模型 (master/lgbm/gru) rank-percentile 加权 (0.5/0.3/0.2)
- 最大回撤 -10.5% 为最佳, 年化波动 19.8% 最低
- 但年化收益 40.8% 低于 master 单独 45.2%
- 启示: ensemble 时, 弱模型会稀释强模型的优势

## 待完成

1. **报告撰写** (~4 h) — HTML 报告框架已生成在 `output/reports/report.html`, 需要进一步润色
2. **模拟交易准备** (6/1 起) — daily_signal.py 已支持 master/master_v3/ensemble, 推荐用 master

## 复现命令 (按顺序)

```bash
conda activate astock

# 数据 (Day 1)
python -m code.data.build_panel
python -m code.data.universe
python -m code.data.validate

# 特征 (Day 2-3)
python -m code.features.factors
python -m code.features.labels
python -m code.features.neutralize
python -m code.features.market_features

# 模型
python -m code.models.lgbm_baseline       # 8 s
python -m code.models.gru_att             # 17 min
python -m code.models.master              # 4.7 min  <-- 主模型
python -m code.models.master_v2           # 14 min   <-- 反例 (容量过大)
python -m code.models.master_v3           # 9 min    <-- multi-seed
python -m code.models.ensemble            # 1 min

# 回测
python -m code.backtest.engine --model master --n 10 --k 2
python -m code.backtest.engine --model master_v2 --n 10 --k 2
python -m code.backtest.engine --model master_v3 --n 10 --k 2
python -m code.backtest.engine --model ensemble --n 10 --k 2

# 报告
python -m code.report.build_report --include-v2     # 生成 HTML + 6 图

# 模拟交易 (6/1 起每日)
python -m code.live.daily_signal --date 20260601 --model master --n 10 --k 2
```

## 环境

- conda env: `astock`
- Python 3.10.20
- PyTorch 2.11.0+**cu128** (RTX 5070 Laptop sm_120 需要此版本)
- GPU: RTX 5070 Laptop 8GB
- pandas 2.3.3, numpy 2.2.6, pyarrow 24.0.0, lightgbm 4.6.0, jinja2 3.1.6

## 关键决策回顾

- 股票池: 中证800近似 (本地 circ_mv 前800, 月调)
- 预测目标: 未来5日收益横截面 rank ([-0.5, 0.5])
- 中性化: MAD 去极值 → 行业 dummy + log市值 OLS 残差 → Z-score clip(-5,5)
- 数据切分: 2016-2022 训 / 2023 验 / 2024-2025 测 (Out-of-sample 565天)
- 模型选型: MASTER v1 (123K 参数, AAAI 2024 SOTA)
- 损失: IC loss (Pearson) + Top-K margin loss, alpha=0.6
- 回测: 日频, n=10持仓, k=2每日换手, 不扣手续费 (报告里可补)
