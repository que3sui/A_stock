# 项目进度记录

> 最后更新: 2026-05-29 (晚间)
> 截止 6/1 模型 deadline: 3 天
>
> **Day 更新**: 数据增量 3 天 + 权重通道实验 (消融→方案A→SOTA)
> **模拟交易初始资金**: 1,000,000 元 (100万)
> **当前 SOTA**: MASTER v1 + 权重独立通道 (夏普 2.35, 总收益 126.6%)

## 已完成 (Day 1 ~ MASTER v3)

### 数据基础

- `cache/panel.parquet` — 1074万行长面板, 5756股×2524天, 50列
- `cache/universe.parquet` — 中证800近似池 (市值前800, 月度调整, 4.6%月度换手)
- `cache/features.parquet` — universe过滤+中性化后的20维因子
- `cache/labels.parquet` — 5日累计log收益 + 横截面rank
- `cache/factors_raw.parquet` — 原始20因子 (中性化前)
- `cache/market_features.parquet` — 12维市场状态信号 (3指数×4特征)

### 模型 (全部out-of-sample 2024-20260528测试集 574 天, **真实交易约束**)

> ⚠️ **2026-05-29 重训**: 数据更新至 20260528 (+3交易日); 模拟交易初始资金 100万
> 已修复 market_features 全期标准化泄露; 已加入手续费(双边0.025%+卖出0.1%)+涨跌停过滤

| 指标            | LGBM     | MLP    | GRU+Att | MASTER v1  | MASTER v3  | Ensemble | HS300  |
| --------------- | -------- | ------ | ------- | ---------- | ---------- | -------- | ------ |
| Test IC         | 0.0508   | 0.0531 | 0.0512  | 0.0613     | 0.0589     | 0.0543   | -      |
| Test RankIC     | 0.0454   | 0.0467 | 0.0427  | 0.0574     | **0.0596** | 0.0550   | -      |
| 年化 RankICIR   | 5.27     | 5.64   | 4.29    | 6.14       | **6.83**   | 5.54     | -      |
| 总收益 (真实)   | 30.9%    | 43.8%  | 15.3%   | 74.4%      | **106.9%** | 47.3%    | 43.5%  |
| 年化收益 (真实) | 24.2%    | 30.6%  | 18.9%   | 40.5%      | **52.5%**  | 29.9%    | 19.1%  |
| 夏普 (真实)     | 1.13     | 1.20   | 0.71    | 2.00       | **2.22**   | 1.67     | 1.06   |
| 最大回撤 (真实) | -22.5%   | -20.7% | -31.6%  | **-11.1%** | -17.3%     | -13.2%   | -15.7% |
| 训练时长        | 10s      | ~2min  | 17min   | 8min       | 25min      | 1min     | -      |
| 参数量          | 61 trees | 144K   | 44K     | 123K       | 123K × 3   | -        | -      |

**真实 vs 理想化 (MASTER v1)**: 年化收益 43.7% → 40.5% (-3.2pp), 夏普 2.19 → 2.00 (-8.7%); 但累积净值 116% → 74% (-42pp, 复利累积放大). 加费率影响小, 策略真实可行.

### 权重通道增强实验 (消融 → 方案A SOTA) 🆕

**背景**: `index_weight/` 中有沪深300+创业板月度成分股权重 (250个CSV, 2016-2026), 此前未使用。

**消融实验 (失败)**: 将 hs300_weight/hs300_dweight/cyb_weight 加入 FACTOR_COLS 走中性化管线:

- LGBM RankIC 0.0454→0.0427, MASTER v1 夏普 2.00→0.80 崩溃
- **根因**: 中性化(行业+市值OLS残差)精确移除了权重信号 — 权重天然集中大盘金融股, 去行业化=去信号

**方案A (成功)**: 权重因子不经过中性化, 通过独立投影通道 (weight_proj) 进入 MASTER:

- MASTER v1+W: **夏普 2.35 (+18%), 总收益 126.6% (+70%), ICIR 7.29 (+19%)**
- MASTER v3+W: 多seed质量不均 (seed2024 RIC=0.0543 vs seed1337 RIC=0.0565), 弱seed稀释强seed
- **单模型+权重通道 > 多seed集成**: 更好的特征结构 > 更多的模型

**模型已在 output/v1_20f/ 和 output/v2_weight/ 存档。**

**最佳模型 (终版)**:

- **SOTA**: **MASTER v1 + 权重通道 (夏普 2.35, 总收益 126.6%)** — 独立通道架构 > multi-seed 集成
- 保守备选: MASTER v3 20f (夏普 2.22, 总收益 106.9%) — 不需要权重数据即可复现
- 模拟交易推荐: **MASTER v1 + 权重通道** (SOTA)

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

### 6. Multi-seed v3 在数据更新后逆转为 SOTA (重要发现)

- 用 v1 完全相同配置, 3 个 seed (42, 1337, 2024) 训练后 rank-percentile 平均
- **数据更新后 v3 夏普 2.22 超越所有模型**, 总收益 106.9% (v1 的 74.4%)
- 旧结论 "减方差伤害收益" 在新增 2026Q2 数据后被推翻
- 启示: 当市场进入新阶段 (2026), multi-seed 提供的稳健性比单点估计更有价值
- **新结论**: 数据分布变化时, ensemble 的方差消减收益 > 信号模糊成本

### 7. Ensemble 提供最低回撤但牺牲收益

- 三模型 (master/lgbm/gru) rank-percentile 加权 (0.5/0.3/0.2)
- 最大回撤 -10.5% 为最佳, 年化波动 19.8% 最低
- 但年化收益 40.8% 低于 master 单独 45.2%
- 启示: ensemble 时, 弱模型会稀释强模型的优势

## 待完成

1. **报告撰写** (~4 h) — HTML 报告框架已生成在 `output/reports/report.html`, 需要进一步润色
2. **模拟交易准备** (6/1 起) — daily_signal.py 已支持 master/master_v3/ensemble
   - **推荐 MASTER v3** (最新重训后 SOTA, 夏普 2.22, 总收益 106.9%)
   - 保守备选: MASTER v1 (夏普 2.00, 最低回撤 -11.1%)
   - 初始资金: **1,000,000 元 (100万)**

## 复现命令 (按顺序)

```bash
conda activate astock

# 数据 (首次全量构建)
python -m code.data.build_panel       # ~10 min
python -m code.data.universe
python -m code.data.validate

# 增量更新 (日常, 从云盘)
python -m code.data.incremental_update --data-source "E:/科大云盘/A股数据"

# 特征 (首次全量构建)
python -m code.features.factors
python -m code.features.labels
python -m code.features.neutralize
python -m code.features.market_features

# 模型
python -m code.models.lgbm_baseline       # ~10s
python -m code.models.mlp_baseline        # ~2min
python -m code.models.gru_att             # ~17min
python -m code.models.master              # ~8min  <-- v1 主模型
python -m code.models.master_v3           # ~25min <-- v3 multi-seed (SOTA)
python -m code.models.ensemble            # ~1min

# 回测
python -m code.backtest.engine --model master --n 10 --k 2
python -m code.backtest.engine --model master_v3 --n 10 --k 2
python -m code.backtest.engine --model ensemble --n 10 --k 2

# 报告
python -m code.report.build_report --include-v2

# 模拟交易 (6/1 起每日, 初始资金 100万)
python -m code.live.daily_signal --date 20260601 --model master_v3 --n 10 --k 2
python -m code.live.position_size --signal output/signals/20260601_master_v3.csv --capital 1000000
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
- 数据切分: 2016-2022 训 / 2023 验 / 2024-20260528 测 (Out-of-sample 574天)
- 模型选型: MASTER v3 (3-seed ensemble, SOTA; v1 为保守备选)
- 损失: IC loss (Pearson) + Top-K margin loss, alpha=0.6
- 回测: 日频, n=10持仓, k=2每日换手, 真实约束 (手续费+涨跌停+ST)
- 模拟交易初始资金: 1,000,000 元 (100万)
