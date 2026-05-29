# 第一版: 20因子基准模型

训练日期: 2026-05-29
因子数: 20 (动量/反转/波动/流动性/资金流/基本面/技术)
数据: 2016-2024Q1训 / 2024Q2-2026Q1验 / 2024Q2-20260528测 (574天)
中性化: MAD去极值 → 行业+市值OLS残差 → Z-score

## 模型结果 (真实交易约束, n=10, k=2)

| 模型 | RankIC | ICIR年化 | 夏普 | 总收益 | 最大回撤 |
|------|--------|---------|------|--------|---------|
| MASTER v1 | 0.0574 | 6.14 | 2.00 | 74.4% | -11.1% |
| MASTER v3 | 0.0596 | 6.83 | 2.22 | 106.9% | -17.3% |

## 文件清单

- `checkpoints/master.pt` — MASTER v1 (seed 42, 122K params)
- `checkpoints/master_seed1337.pt` — MASTER v1 (seed 1337)
- `checkpoints/master_seed2024.pt` — MASTER v1 (seed 2024)
- `signals/master_test.parquet` — MASTER v1 测试集预测
- `signals/master_v3_test.parquet` — MASTER v3 三seed集成预测
