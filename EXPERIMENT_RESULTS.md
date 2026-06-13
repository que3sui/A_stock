# 实验结果数据说明

> 面向撰写实验报告的组员，说明每个实验结果文件的用途、格式和使用方式。
> 原始数据（daily/metric/moneyflow 等）的说明见 `DATA_README.md`。

---

## 快速索引

| 你要找什么                                         | 文件位置                                                   |
| -------------------------------------------------- | ---------------------------------------------------------- |
| 各模型回测指标对比（Sharpe / 年化收益 / 最大回撤） | `output/backtest_*_real_metrics.json`                      |
| 净值曲线汇总对比图                                 | `output/reports/figs/nav_compare.png`                      |
| 各模型单独回测图                                   | `output/reports/backtest_*.png`                            |
| 训练损失曲线                                       | `output/reports/figs/loss_curves.png`                      |
| IC 时序分析                                        | `output/reports/figs/ic_by_year.png` / `ic_by_quarter.png` |
| 超参搜索结果                                       | `output/v3_multi_train/search_summary.json`                |
| 完整报告（可直接引用）                             | `深度学习基础大作业.pdf`                                   |

---

## 一、回测指标 — `output/backtest_*_metrics.json`

### 文件列表

| 文件名                                 | 模型                           | 约束类型                      |
| -------------------------------------- | ------------------------------ | ----------------------------- |
| `backtest_master_real_metrics.json`    | MASTER v1                      | 真实（T+1 + 涨跌停 + 手续费） |
| `backtest_master_v3_real_metrics.json` | **MASTER v3（SOTA）**          | 真实                          |
| `backtest_master_v7_real_metrics.json` | MASTER v7（Sharpe-aware loss） | 真实                          |
| `backtest_lgbm_real_metrics.json`      | LightGBM                       | 真实                          |
| `backtest_gru_real_metrics.json`       | GRU+Attention                  | 真实                          |
| `backtest_mlp_real_metrics.json`       | MLP                            | 真实                          |
| `backtest_ensemble_real_metrics.json`  | Ensemble（LGBM+GRU+MLP投票）   | 真实                          |
| `backtest_*_ideal_metrics.json`        | 各模型对应                     | 理想化（无手续费+无涨跌停）   |
| `backtest_master_v1raw_metrics.json`   | MASTER v1 原始（无权重通道）   | 消融参考                      |

### JSON 结构

```json
{
  "strategy": {
    "sharpe": 2.4664,           // 年化夏普比率
    "annual_return": 0.5680,    // 年化收益率
    "max_drawdown": -0.1577,    // 最大回撤
    "calmar": 3.6027,           // Calmar 比率 (年化收益/|最大回撤|)
    "win_rate": 0.5320,         // 日胜率
    "total_return": 1.2254,     // 累计收益率
    "volatility": 0.2302,       // 年化波动率
    "mean_daily_ret": 0.0018,   // 日均收益
    "daily_returns": [...],     // 每日收益序列
    "nav": [...]                // 净值序列
  },
  "benchmark": { ... },         // 中证800等权基准（同上结构）
  "excess_annual": 0.3727,      // 年化超额收益
  "n_holdings": 10,             // 持仓股票数
  "k_swap": 2,                  // 每期最大换手数
  "start_date": 20240101,       // 回测起始日
  "apply_fee": true,
  "apply_limit": true
}
```

### 核心对比表（n=10, k=2, 真实约束, 2024-01 ~ 2026-05）

| 模型                 | Sharpe   | 年化收益  | 最大回撤   | Calmar   | 超额年化  |
| -------------------- | -------- | --------- | ---------- | -------- | --------- |
| **MASTER v3 (SOTA)** | **2.47** | **56.8%** | **-15.8%** | **3.60** | **37.3%** |
| MASTER v7            | 1.90     | 43.7%     | -17.5%     | 2.51     | 24.2%     |
| MASTER v1            | 1.61     | 41.4%     | -22.2%     | 1.86     | 21.9%     |
| LightGBM             | 1.30     | 25.3%     | -15.5%     | 1.64     | 6.3%      |
| Ensemble             | 1.08     | 24.7%     | -21.8%     | 1.13     | 5.7%      |
| MLP                  | 0.96     | 20.5%     | -19.5%     | 1.05     | 1.4%      |
| GRU+Att              | 0.92     | 23.9%     | -28.8%     | 0.83     | 4.9%      |

> MASTER v3 是 multi-seed 训练搜索的 Top-3 种子（314, 1024, 1337）的 rank-percentile 平均集成。

---

## 二、回测净值曲线 — `output/reports/`

### 各模型回测图

命名规则: `backtest_{模型}_{约束}.png`

| 文件                          | 内容                                   |
| ----------------------------- | -------------------------------------- |
| `backtest_master_real.png`    | MASTER v1 策略净值 vs 基准，附回撤曲线 |
| `backtest_master_v3_real.png` | MASTER v3 (SOTA)                       |
| `backtest_master_v7_real.png` | MASTER v7 (Sharpe-aware loss)          |
| `backtest_lgbm_real.png`      | LightGBM                               |
| `backtest_gru_real.png`       | GRU+Attention                          |
| `backtest_mlp_real.png`       | MLP                                    |
| `backtest_ensemble_real.png`  | Ensemble                               |
| `backtest_*_ideal.png`        | 对应理想化回测（无手续费+无涨跌停）    |

### 对应 NAV 数据

`backtest_*_nav.csv` — 日频净值序列，可用 Excel / Matplotlib 自行绘图。

---

## 三、分析图表 — `output/reports/figs/`

| 文件                        | 内容                                       | 建议用途     |
| --------------------------- | ------------------------------------------ | ------------ |
| `nav_compare.png`           | 所有模型净值曲线汇总对比                   | **主对比图** |
| `ic_by_year.png`            | 年度 Rank IC（MASTER + LightGBM）          | IC 分析      |
| `ic_by_quarter.png`         | 季度 Rank IC                               | IC 分析      |
| `ic_cumulative.png`         | 累计 Rank IC                               | IC 分析      |
| `loss_curves.png`           | MASTER v1/v2 训练损失曲线                  | 训练分析     |
| `loss_curves_v3.png`        | v3 multi-seed 训练损失曲线                 | 训练分析     |
| `sensitivity_nk.png`        | N/K 参数敏感性（不同持仓/换手下的 Sharpe） | 参数分析     |
| `drawdown.png`              | 各模型回撤曲线对比                         | 风险分析     |
| `quantile_master.png`       | 分位数组合收益（检验因子单调性）           | 因子检验     |
| `monthly_returns.png`       | 月度收益热力图                             | 收益分解     |
| `factor_importance.png`     | 因子重要性排序                             | 特征分析     |
| `attribution_industry.png`  | 行业暴露归因                               | 归因分析     |
| `attribution_marketcap.png` | 市值暴露归因                               | 归因分析     |
| `data_flow.png`             | 数据管线流程图                             | 方法论       |
| `master_arch.png`           | MASTER 模型架构图                          | 方法论       |

---

## 四、超参搜索 — `output/v3_multi_train/`

### `search_summary.json`

10-seed 训练搜索结果汇总，每个种子包含:

- `val_rank_ic` / `test_rank_ic` — 验证集/测试集 Rank IC
- `test_rank_icir_annual` — 年化 ICIR
- `sharpe` — 单独回测 Sharpe
- `total_return` / `max_drawdown`
- `train_time_s` — 训练耗时

**最佳种子**: 1024（val_rank_ic=0.0675, sharpe=2.47）

### `all_seeds_metrics.json`

11 个种子的 IC 指标汇总，按 `test_rank_ic` 降序排列。

| 种子 | test_rank_ic | 年化 ICIR |
| ---- | ------------ | --------- |
| 4096 | 0.0629       | 7.71      |
| 777  | 0.0585       | 7.76      |
| 88   | 0.0579       | 6.85      |
| 128  | 0.0573       | 6.80      |
| 99   | 0.0572       | 6.91      |
| 1024 | 0.0565       | 6.83      |
| 42   | 0.0560       | 7.42      |
| 1337 | 0.0518       | 6.37      |
| 2048 | 0.0512       | 6.67      |
| 314  | 0.0507       | 6.15      |
| 2024 | 0.0494       | 6.56      |

---

## 五、辅助分析数据 — `output/`

| 文件                               | 内容                   |
| ---------------------------------- | ---------------------- |
| `sensitivity_nk.json` / `.csv`     | N/K 参数敏感性原始数据 |
| `direction_accuracy.json`          | 方向预测准确率         |
| `attribution.json`                 | 行业/市值归因原始数据  |
| `ic_segment_metrics.json`          | IC 按市场状态分段统计  |
| `data_audit.json`                  | 数据管线校验结果       |
| `*_metrics.json` (不含 backtest\_) | 各模型训练阶段 IC 指标 |

---

## 六、报告 PDF

`深度学习基础大作业.pdf` — 课程大作业终稿，章节结构:

1. 绪论（问题定义）
2. 模型与对比（含 OOS 综合指标对比表）
3. 结果分析（净值曲线 / IC / 分位数 / 回撤 / 因子重要性 / 归因）
4. 消融实验与架构迭代
5. 模拟交易实盘
6. 实验总结

可直接从中引用图表和表格数据。

---

## 七、自行复现

```bash
conda activate astock

# 回测（生成 backtest_*_metrics.json + backtest_*.png）
python -m code.backtest.engine --model master --n 10 --k 2

# 完整报告（生成 output/reports/ 下所有图表）
python -m code.report.build_report --include-v2

# IC 分析
python -m code.report.ic_analysis

# 超参搜索（生成 output/v3_multi_train/）
python -m code.models.master_train_search
```

---

## 附注

- 回测测试期: **2024-01-01 ~ 2026-05-29**，基准: 中证800等权
- 真实约束: T+1 交易 + 涨跌停过滤 + 双边手续费（买 0.025% 佣金 + 卖 0.025% 佣金 + 0.1% 印花税）
- 模型权重文件（`.pt`, `.pkl`）过大不在此仓库，需联系训练者
- 原始数据（daily/metric/moneyflow）通过 symlink 指向科大云盘，说明见 `DATA_README.md`
