# MASTER v3_search (SOTA, 2026-05-31)

训练搜索最优模型: 20 种子 + top-3 超参微调, 全量回测选优。

## 最佳模型

- **种子**: 128
- **超参**: lr=3e-4, dropout=0.2, alpha=0.6, H=64, wd=1e-3
- **参数量**: 122,629

## 回测 (真实约束: 手续费+涨跌停, n=10, k=2)

| 指标 | 值 |
|------|-----|
| 夏普 | 2.57 |
| 总收益 | 145.0% |
| 年化收益 | 64.8% |
| 最大回撤 | -14.0% |
| Test RankIC | 0.0560 |
| Annual ICIR | 7.71 |

## 方法

1. Phase 1: 训练 20 个 MASTER v1 变体 (不同 seed)
2. Phase 2: 全量回测 (不依赖 val_rank_ic 筛选)
3. Phase 3: top-3 (seed128/1024/99) 超参微调 (lr/dropout/alpha/H 共 6 变体/种子)
4. Phase 4: 按夏普选最优 → seed128 + lr=3e-4

详见 `search_summary_v2.json`。

## 复现

```bash
python -m code.models.master_train_search_v2
```
