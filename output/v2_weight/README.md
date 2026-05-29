# 第二版: 权重通道增强模型 (SOTA)

训练日期: 2026-05-29
因子: 20中性化 + 3权重穿透通道 (hs300_weight/hs300_dweight/cyb_weight)
架构: MASTER with dual-channel input (stock_proj 20→48 + weight_proj 3→16 = concat 64)
数据: 同第一版

关键改进: 指数权重因子不经过行业+市值中性化, 通过独立投影通道进入MASTER,
保留原始横截面信号, 避免中性化"去行业化"消灭权重因子的核心alpha。

## 模型结果 (真实交易约束, n=10, k=2)

| 模型 | RankIC | ICIR年化 | 夏普 | 总收益 | 最大回撤 |
|------|--------|---------|------|--------|---------|
| MASTER v1 + 权重 | 0.0549 | 7.29 | 2.35 | 126.6% | -12.5% |

## vs 第一版提升

| 指标 | 第一版 v1 | 第二版 v1 | 变化 |
|------|----------|----------|------|
| 夏普 | 2.00 | 2.35 | +18% |
| 总收益 | 74.4% | 126.6% | +70% |
| 年化收益 | 40.5% | 59.4% | +47% |
| ICIR年化 | 6.14 | 7.29 | +19% |

## 文件清单

- `checkpoints/master.pt` — MASTER v1 + 权重通道 (SOTA, 122K params)
- `checkpoints/master_seed1337.pt` — 权重通道 seed 1337
- `checkpoints/master_seed2024.pt` — 权重通道 seed 2024
- `signals/master_test.parquet` — 测试集预测
- `signals/master_v3_test.parquet` — V3 三seed集成 (多seed质量不均, 不如v1单模型)
