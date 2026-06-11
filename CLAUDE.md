# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

A股趋势预测与模拟交易系统 — 科大深度学习基础大作业 (2026)。预测未来5日横截面 rank-percentile，主模型为 MASTER (Market-guided Stock Transformer, AAAI 2024)，含时序+横截面双轴注意力 + 独立权重通道。

## 环境

- conda env: `astock`，Python 3.10.20
- PyTorch 2.11.0+**cu128**（RTX 5070 Laptop sm_120 必须此版本，其他显卡按官网选）
- 显式 Python 路径: `/d/anaconda3/envs/astock/python`（base 环境缺 CUDA/lightgbm）
- 关键依赖: lightgbm 4.6.0, transformers 5.9.0, pandas 2.3.3, pyarrow 24.0.0

## 常用命令

```bash
# 数据（首次全量 / 日常增量）
/d/anaconda3/envs/astock/python -m code.data.build_panel          # ~10 min，整合 daily/metric/moneyflow → panel.parquet
/d/anaconda3/envs/astock/python -m code.data.universe             # 中证800近似池
/d/anaconda3/envs/astock/python -m code.data.validate             # 6项数据正确性检查
/d/anaconda3/envs/astock/python -m code.data.incremental_update --data-source "E:/科大云盘/A股数据"

# 特征工程
/d/anaconda3/envs/astock/python -m code.features.factors           # 20个原始因子
/d/anaconda3/envs/astock/python -m code.features.labels            # 5日forward log return + 横截面rank
/d/anaconda3/envs/astock/python -m code.features.neutralize        # MAD + 行业/市值中性化 + Z-score
/d/anaconda3/envs/astock/python -m code.features.market_features   # 12维市场状态（train段标准化，防泄露）

# 模型训练
/d/anaconda3/envs/astock/python -m code.models.lgbm_baseline       # ~52s, 83 trees
/d/anaconda3/envs/astock/python -m code.models.mlp_baseline        # MLP基线
/d/anaconda3/envs/astock/python -m code.models.gru_att             # ~18.5min
/d/anaconda3/envs/astock/python -m code.models.master              # v1, ~9.6min
/d/anaconda3/envs/astock/python -m code.models.master_v3           # v3 multi-seed, ~8.3min
/d/anaconda3/envs/astock/python -m code.models.ensemble            # <1min

# v4-v7 实验模型（输出到 output/v{N}/）
/d/anaconda3/envs/astock/python -m code.models.master_v4 --search  # 时间衰减+行业轮动+训练搜索
/d/anaconda3/envs/astock/python -m code.models.master_v7 --search  # Sharpe-aware loss

# v3_search (SOTA, 10-seed搜索, ~15min, 输出到 output/v3_multi_train/)
/d/anaconda3/envs/astock/python -m code.models.master_train_search

# v3_search 信号生成 (N=5集中持仓, 内联脚本)
# checkpoint 在 output/v3_multi_train/checkpoints/, seeds=[314,1024,1337]
# 如 daily_signal.py 已添加 --model v3_search 支持则直接:
/d/anaconda3/envs/astock/python -m code.live.daily_signal --date 20260604 --model master_v3 --n 5 --k 2

# 回测（真实约束: T+1 + 涨跌停 + 双边手续费）
/d/anaconda3/envs/astock/python -m code.backtest.engine --model master --n 10 --k 2
/d/anaconda3/envs/astock/python -m code.backtest.engine --model master --n 10 --k 2 --no-fee --no-limit  # 理想化对比

# 模拟交易（每日盘后，初始资金100万）
/d/anaconda3/envs/astock/python -m code.live.daily_signal --date 20260602 --model master --n 10 --k 2
/d/anaconda3/envs/astock/python -m code.live.position_size --signal output/signals/20260602_master.csv --capital 1000000

# 报告
/d/anaconda3/envs/astock/python -m code.report.build_report --include-v2
/d/anaconda3/envs/astock/python -m code.report.ic_analysis
/d/anaconda3/envs/astock/python -m code.report.loss_curves

# 环境验证
/d/anaconda3/envs/astock/python -m code.check_env
```

## 数据管线流程

```
daily/ + metric/ + moneyflow/ + stock_st/
  → build_panel.py → panel.parquet (1074万行, 5756股×2524天, 50列)
  → universe.py → universe.parquet (中证800近似, 月调)
  → factors.py → factors_raw.parquet (20因子)
  → neutralize.py → features.parquet (MAD去极值→行业dummy+log市值OLS残差→Z-score clip(-5,5))
  → labels.py → labels.parquet (5日累计log收益 + rank)
  → market_features.py → market_features.parquet (13维: 12维市场+news_count, train段mean/std标准化)
```

**数据切分**: 2016-2022 训练 / 2023 验证 / 2024-20260529 测试（严格按时间，无随机打乱）

**股票池**: 中证800近似（每月按 circ_mv 前800，本地计算，月换手约4.6%）

## 共享模块（重构后新增）

| 模块 | 用途 |
|------|------|
| `code/config.py` | **单一数据源**: FACTOR_COLS, WEIGHT_COLS, T, TRAIN_MAX, ROOT/CACHE/OUTPUT, MASTER_* 超参数 |
| `code/losses.py` | `ic_loss()` / `topk_margin_loss()` / `combined_loss()` — 消除 4 处重复 |
| `code/metrics.py` | `daily_ic()` / `ic_summary()` — 消除 4 处重复 IC 聚合 |

**使用约定:** 所有文件的 `FACTOR_COLS`、`WEIGHT_COLS`、`ROOT`/`CACHE`/`OUTPUT` 必须从 `code.config` 导入，不得本地定义。

## 架构核心

### MASTER 模型设计

```
输入: X[N,T,20] 因子 + X_w[N,T,3] 权重通道 + market[T,13] 市场状态
  → Market-Gate: 市场状态调制因子
  → Intra-stock Transformer (时序 axis, 每只股票内部)
  → Inter-stock Transformer (横截面 axis, 股票之间)
  → 输出: scores[N] (横截面 rank-percentile 预测)

损失: IC loss (Pearson) + Top-K margin loss, default alpha=0.6
By-day batch: 每个batch是一天所有股票，直接对横截面排序
T=20 (回溯20天), H=64, ~123K参数
```

### 两通道设计（关键架构决策）

- **中性化通道 (20因子)**: 行业+市值OLS残差后的纯alpha信号，走主pipeline
- **权重通道 (3因子)**: hs300_weight / hs300_dweight / cyb_weight，**不经过中性化**，通过独立 weight_proj 进入MASTER。中性化会精确移除权重中的大盘/金融股信号导致模型崩溃（夏普 2.00→0.80）

### 模型版本演进

| 版本 | 文件 | 要点 | 状态 |
|------|------|------|------|
| v1 | `master.py` | 基础MASTER + 权重通道 | 基线 |
| v2 | `master_v2.py` | 加深加宽→过拟合（反例） | 反例 |
| v3 | `master_v3.py` | 3-seed rank-percentile平均 | 已废弃 |
| **v3_search** | `master_train_search.py` | **10-seed搜索 + Top-3集成，当前SOTA** | **Sharpe 2.47** |
| v4 | `master_v4.py` | 时间衰减loss + 行业轮动 + 动态仓位 | 实验 |
| v5 | `master_v5.py` | v4 + 滤波器因子 | 实验 |
| v6 | `master_v6.py` | v5 + 长窗口 | 实验 |
| v7 | `master_v7.py` | Sharpe-aware loss | Sharpe 1.90 |

**当前SOTA:** `master_train_search.py` 输出到 `output/v3_multi_train/`，最佳种子 314/1024/1337 集成，回测 Sharpe 2.47 (n=10, k=2, 真实约束)。

v1-v3 输出到 `output/checkpoints/` 和 `output/signals/`；v4-v7 输出到 `output/v{N}/`；v3_search 输出到 `output/v3_multi_train/`。

### 回测引擎 (`code/backtest/engine.py`)

- T+1隐式：portfolio来自上日score，用当日pct_chg计收益
- 涨跌停过滤：一字板(open==high==low)自动跳过
- 双边手续费：买入0.025%佣金 + 卖出0.025%佣金+0.1%印花税
- ST当日动态剔除（stock_st/数据）
- 支持 `--model lgbm/mlp/gru/master/master_v2/master_v3/master_v7/ensemble`

## 测试与质量门禁

对 `code/models/` 下的任何 Python 模型文件（尤其是 `v*_*.py`）进行修改后，必须：

1. 运行 `python -m code.check_env` 验证环境正常后再报告完成
2. 在输出信号中验证无 NaN/Inf：打印 `scores.isnan().sum()` 和 `scores.isinf().sum()` 后再呈现结果
3. 确认加载的 checkpoint 版本与预期一致：在日志中打印模型版本号 + 时间戳
4. 在 bash 脚本中**绝不使用 `set -e`** 而不进行显式错误处理——优先使用 `|| true` 或 trap 清理

**原因:** 跨会话统计显示 29 次 "代码有 Bug" 事件；NaN 信号、加载错误 checkpoint、`set -e` 崩溃以及未测试的变更是反复出现并耗费数小时迭代的根因。

## 模型版本约定

- 所有模型遵循命名规范: `master_{descriptor}.py`（如 `master_train_search.py` 为 SOTA、`master_v7.py` 为 Sharpe-aware loss）
- 始终在 `MODEL_REGISTRY.md` 中记录: 版本、Sharpe ratio、训练日期、特征集
- **v3_search（Sharpe ~2.47）是 SOTA 基线**；任何新模型必须在相同测试期上击败 v3_search 才能被视为改进
- 当复杂度增加导致性能退化时，回退到更简单的架构并在 MODEL_REGISTRY.md 中记录原因
- v4-v7 全部低于 v3_search——这是一个系统性教训：先验证简单方案，再叠加复杂度

## 关键约定与陷阱

### 代码约定

- 所有脚本用 `python -m code.xxx.yyy` 运行（模块路径），不使用 `python code/xxx/yyy.py`
- `ROOT`/`CACHE`/`OUTPUT` 从 `code.config` 导入，不要本地定义 `ROOT = Path(__file__).resolve().parents[2]`
- `cache/` 存中间产物（panel/features/labels），`output/` 存最终产出（checkpoints/signals/reports）
- 数据源通过 symlink 指向云盘: `daily/ → /e/科大云盘/A股数据/daily/`
- **调仓逻辑** (daily_signal.py): 卖出按得分升序最多 K 只 + 买入填满到 N 只（总换手 ≤ K），僵尸持仓无条件卖

### 已知陷阱

- **FACTOR_COLS 已集中化**: 所有定义统一在 `code/config.py`，其他文件必须 `from code.config import FACTOR_COLS`，不得本地定义
- **Market features 实际 13 维**（含 `news_count`），不是 12 维。推理时必须 `F_market=len(market_cols)` 动态读取
- **IC ≠ 回测收益**（已反复验证3次）：IC是全样本相关性，不度量top-K选股质量
- **权重因子不能走中性化管线**：行业+市值OLS残差会精确移除权重信号
- **market_features 必须用train段mean/std标准化**：全期标准化 = 数据泄露（已修复过一次）
- **早期日期 market 数据越界**: v1 已回移植 v2 的 `market_date_idx[d] >= T - 1` guard（2026-06-05 修复）
- **Windows GBK编码**: tqdm进度条特殊字符可能崩溃，必要时 `2>/dev/null`
- **数据同步**: daily/、market/、metric/、moneyflow/、stock_st/ 需全部同步，缺一会导致静默数据不对齐

### 数据泄露自查清单

- [x] 因子按日横截面中性化（无未来均值/方差）
- [x] market_features用train段(≤2022) mean/std标准化
- [x] 数据集严格按时间切分，无随机打乱
- [x] universe每月用当月可见circ_mv重新选（无回顾性偏差）
- [x] 回测中ST用当日stock_st名单动态剔除
- [x] portfolio来自上一日score → 当日pct_chg计收益（隐式T+1）
- [x] 标签时间(t+1~t+5)严格晚于特征时间(≤t)
