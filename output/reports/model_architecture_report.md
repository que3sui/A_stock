# 模型架构报告：A股趋势预测与横截面选股

## 1. 问题形式化定义

### 1.1 任务描述

给定 $t$ 日横截面上 $N$ 只股票，每只股票有过去 $T=20$ 个交易日的 $F=20$ 个量价因子，预测该股票在未来5个交易日的累计对数收益在横截面上的 **rank-percentile**（排名分位数）。

形式化：

$$
\begin{aligned}
\text{输入:}&\quad \mathbf{X} \in \mathbb{R}^{N \times T \times F},\quad \mathbf{X}^w \in \mathbb{R}^{N \times T \times 3},\quad \mathbf{m} \in \mathbb{R}^{T \times 12} \\
\text{输出:}&\quad \hat{\mathbf{s}} \in \mathbb{R}^N \\
\text{目标:}&\quad \mathbf{y} = \text{rank\_pct}\left(\frac{P_{t+5} - P_{t+1}}{P_{t+1}}\right) \in [0, 1]^N
\end{aligned}
$$

### 1.2 为什么是 rank-percentile 而非回归/分类？

| 方案            | 问题                                                 |
| --------------- | ---------------------------------------------------- |
| 直接回归收益率  | 收益率的绝对量级受市场beta主导，无法区分alpha        |
| 涨跌二分类      | 丢失排序信息，无法选出"最好 vs 较好"                 |
| **横截面 rank** | 天然适配选股：每天只需知道谁比谁好，不受市场涨跌干扰 |

### 1.3 损失函数

MASTER v1 使用双目标联合损失（$\alpha=0.6$）：

**IC Loss**（Pearson 相关系数的负值，驱动全横截面对齐）：

$$\mathcal{L}_{\text{IC}} = -\frac{\sum_i (\hat{s}_i - \bar{\hat{s}})(y_i - \bar{y})}{\sqrt{\sum_i (\hat{s}_i - \bar{\hat{s}})^2 \sum_i (y_i - \bar{y})^2}}$$

**Top-K Margin Loss**（确保预测的 top-K 真实表现优于 bottom-K）：

$$\mathcal{L}_{\text{margin}} = \frac{1}{K^2}\sum_{i \in \text{top}}\sum_{j \in \text{bot}} \max(0,\ \hat{s}_j - \hat{s}_i + 0.1)$$

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{IC}} + (1-\alpha) \cdot \mathcal{L}_{\text{margin}}$$

### 1.4 数据切分

| 阶段 | 时间范围          | 样本量    |
| ---- | ----------------- | --------- |
| 训练 | 2016-01 ~ 2022-12 | ~1,500 天 |
| 验证 | 2023-01 ~ 2023-12 | ~240 天   |
| 测试 | 2024-01 ~ 2026-05 | ~576 天   |

严格按时间切分，无随机打乱。股票池为中证800近似（每月按流通市值选前800只，月换手率 ~4.6%）。

---

## 2. 模型谱系总览

### 2.1 全部模型一览

| 模型            | 参数量   | 训练时间 | Test RankIC | RankICIR(年化) | 回测夏普(真实) | 类型              |
| --------------- | -------- | -------- | ----------- | -------------- | -------------- | ----------------- |
| **LGBM**        | 83 trees | ~52s     | 0.0449      | 5.27           | —              | 树模型基线        |
| **MLP**         | 143,873  | ~2min    | 0.0467      | 5.64           | —              | DL 最简基线       |
| **GRU-Att**     | 43,650   | ~18.5min | 0.0428      | 4.27           | —              | 时序 RNN          |
| **MASTER v1** ★ | 122,629  | ~9.6min  | 0.0544      | 7.29           | **1.68**       | Transformer       |
| MASTER v2       | 421,653  | ~15min   | 0.0442      | 5.62           | —              | 加深加宽(过拟合)  |
| **MASTER v3**   | 3×122K   | ~24.9min | 0.0570      | 7.06           | —              | 3-seed 集成       |
| MASTER v4       | ~122K    | ~20min   | —           | —              | 1.74           | 时间衰减+行业轮动 |
| MASTER v7       | ~122K    | ~20min   | —           | —              | **2.38**       | Sharpe-aware loss |
| **Ensemble**    | —        | <1min    | 0.0540      | 6.21           | 1.08           | 三模型加权        |

> ★ = 当前生产模型（模拟交易使用 v1）
> v5/v6 为中间实验版本，指标与 v4/v7 接近但不具独立代表性。

### 2.2 关键发现：IC ≠ 回测收益

这是本项目中反复验证的核心洞察：

- MASTER v1 RankIC = 0.0544 → 回测夏普 1.68
- Ensemble RankIC = 0.0540（几乎相同）→ 回测夏普仅 1.08

IC 是全样本的 Pearson 相关性，衡量的是"横截面上所有股票的排序质量"；而回测收益只取决于"选出的 top-K 股票的实际表现"。两者之间的 gap 解释：

1. **IC 对中段股票敏感**：模型在中段（排名 200-600）的排序提升会显著拉高 IC，但对 top-10 选股无益
2. **尾部非线性**：top-1% 和 top-5% 的收益差远大于 median 附近的收益差，IC 不捕捉这种非对称性
3. **交易约束**：T+1、涨跌停、手续费等摩擦成本在 IC 中不可见

v7 的 Sharpe-aware loss 正是针对这一 gap 的直接回应。

---

## 3. MASTER 架构详解

MASTER（Market-guided Stock Transformer）的设计灵感来自 AAAI 2024 论文，核心思想是 **同时建模时序依赖（intra-stock）和横截面交互（inter-stock）**，并用市场状态信号调制因子表达。

### 3.1 架构总览

```
X [N,T,20] ──→ [Market Gate] ──→ [Stock Proj] ──┐
                 (m modulates X)                    ├──→ [PosEnc] ──→ [Intra Tx] ──→ [Temp Attn] ──→ [Inter Tx] ──→ [Head] ──→ scores[N]
X_w [N,T,3] ──────────────────────→ [Weight Proj] ─┘
```

### 3.2 逐模块解析

#### (a) Market-Gate 门控

```python
m_emb = market_proj(market)          # [T, H]  市场状态嵌入
gate = sigmoid(gate_proj(m_emb))     # [T, F]  逐因子门控值
X = X * gate                         # 按市场状态缩放因子
```

**直觉**：动量因子在牛市中重要、在震荡市中不重要；波动率因子在市场恐慌时重要、在平稳时不重要。Gate 让模型学会了"在不同市场环境下信任不同因子"。

市场状态 $\mathbf{m} \in \mathbb{R}^{T \times 12}$ 包含12个宏观指标：指数收益、波动率、涨跌比、成交额变化、行业离散度等，用训练段（≤2022）的 mean/std 标准化以防止数据泄露。

#### (b) 双通道设计（关键架构决策）

```
中性化因子(20维) → stock_proj → H_stock=48 ─┐
                                              ├→ concat → H=64
权重因子(3维)   → weight_proj → H_weight=16 ─┘
```

- **中性化通道**（20因子）：经过 MAD 去极值 → 行业+市值 OLS 残差 → Z-score clip(-5,5) 的标准管线
- **权重通道**（3因子：hs300_weight, hs300_dweight, cyb_weight）：**不经过中性化**，通过独立的 `weight_proj` 进入模型

**消融证据**：移除权重通道后夏普从 2.00 暴跌至 0.80。原因：行业+市值 OLS 残差会精确移除大盘/金融股的权重信号（这些股票在指数中权重高、市值大），导致模型"看不见"最重要的标的。

#### (c) Intra-Stock Transformer（时序注意力）

对每只股票的 $T=20$ 天序列做 self-attention，学习单个股票内部的时序模式（如趋势启动、动量衰减、波动率聚集）。

- 2层 TransformerEncoder，4头注意力
- 隐藏维度 H=64，FFN=128
- GELU 激活，Dropout=0.2
- 位置编码：正弦位置编码

#### (d) Temporal Aggregation（时序聚合）

```python
temp_query = learnable [1, H]       # 全局可学习查询向量
Z_t = MultiheadAttention(temp_query, Z, Z)  # 注意力池化
```

不简单地取最后一天或平均，而是用一个**可学习的 query 向量**对20天做注意力加权。模型自己学会哪些时间步更重要（近期通常权重更高，但关键拐点处模型会关注远处）。

#### (e) Inter-Stock Transformer（横截面注意力）

将 $N$ 只股票的时序聚合向量 $Z_t \in \mathbb{R}^{N \times H}$ 视为一个序列（batch=N, feature=H），做横截面 self-attention。

**这是本架构最核心的创新**：传统模型（LGBM/MLP/GRU）独立处理每只股票，不知道"今天市场上还有其他什么选择"。Inter-stock Transformer 让模型在排序时能感知股票之间的相对关系——类似基金经理同时看所有备选标的来排优先级。

#### (f) 残差连接 + 输出头

```python
Z_final = Z_t + Z_c              # 残差: 保留时序信息 + 横截面信息
score = LayerNorm → Linear(32) → GELU → Dropout → Linear(1)
```

### 3.3 By-Day Batch 训练

与常规 ML 的随机采样 batch 不同，MASTER 使用 **by-day batch**：每个 batch 是某一天横截面上所有股票（~800只）。这保证了：

- 横截面 loss（IC/Top-K）在同一 batch 内有意义
- 模型学习的就是每天的横截面排序，与推理时的任务完全一致（train-test alignment）

---

## 4. 关键设计决策与消融证据

### 4.1 权重双通道：必需

| 配置                               | 回测夏普 |
| ---------------------------------- | -------- |
| 全因子中性化（权重也被中性化）     | ~0.80    |
| 双通道（权重独立进入，不经中性化） | **2.00** |

结论：中性化管线会精确移除权重信号。权重因子携带的是"哪些股票对指数走势重要"的信息，这个信息本身是有效的 alpha 来源，不应被 OLS 残差消除。

### 4.2 模型容量：v1 > v2（反直觉）

| 版本 | H   | 层数             | T   | 参数量 | RankIC     |
| ---- | --- | ---------------- | --- | ------ | ---------- |
| v1   | 64  | intra=2, inter=1 | 20  | 122K   | **0.0544** |
| v2   | 96  | intra=3, inter=2 | 30  | 421K   | 0.0442     |

v2 加深加宽后验证 IC 更高（0.066 vs 0.053），但测试 IC 反而更低——典型过拟合。A 股 alpha 信号的信噪比极低（IC ~0.05 意味着 $R^2 \approx 0.25\%$），更大模型只是在拟合噪声。这是整个项目最重要的架构教训。

### 4.3 Multi-Seed 集成：稳健但有代价

v3 用 3 个不同 seed 训练的 v1 模型做 rank-percentile 平均：

- RankIC = 0.0570（vs v1 的 0.0544，+4.7%）
- RankIC 年化 = 7.06（vs 7.29，略低因为 std 更大）

multi-seed 在数据分布变化时更稳健（2026 Q1-Q2 的市场风格切换中 v3 表现优于 v1），但训练成本 3 倍。

### 4.4 损失函数演进

| 版本 | 损失函数               | 动机              |
| ---- | ---------------------- | ----------------- |
| v1   | IC + Top-K margin      | 全排序 + 头部区分 |
| v4   | 时间衰减 IC + Top-K    | 近期样本更重要    |
| v7   | **Sharpe-approximate** | 直接优化回测目标  |

v7 的 Sharpe-aware loss 直接定义在 top-K 收益的 mean/std 上（模拟回测 Sharpe），是目前最优方案（夏普 2.38）。

---

## 5. Ensemble 方案

三模型 rank-percentile 加权集成：

$$\hat{s}^{\text{ens}} = 0.5 \cdot \text{rank}(\hat{s}^{\text{master}}) + 0.3 \cdot \text{rank}(\hat{s}^{\text{lgbm}}) + 0.2 \cdot \text{rank}(\hat{s}^{\text{gru}})$$

- 各模型先做日度横截面 rank-percentile 标准化（统一到 [0, 1]）
- 缺失模型（某只股票不在某模型的 universe 内）用当日均值填充
- IC 稳定性提升（RankICIR 6.21），但回测收益反而不如单独 MASTER v1

**推测原因**：LGBM 和 GRU 的引入增加了"共识性"选股——集成倾向于选三个模型都看好的股票，但这往往是大盘蓝筹、beta 暴露高，alpha 纯度反而下降。

---

## 6. 因子体系

### 6.1 20个量价因子

| 类别   | 因子                                      | 说明                                   |
| ------ | ----------------------------------------- | -------------------------------------- |
| 动量   | mom_5, mom_20, mom_60, mom_120            | 多周期收益动量                         |
| 反转   | rev_1, rev_5                              | 短期反转效应                           |
| 波动   | vol_20, vol_60                            | 历史波动率                             |
| 流动性 | turnover_20, amihud_20                    | 换手率 + Amihud 非流动性               |
| 资金流 | mf_net_5, mf_lg_strength, mf_elg_strength | 主力/特大单资金流向                    |
| 估值   | pe_ttm_rank, pb_rank, circ_mv_log         | 市盈率/市净率横截面排名 + 对数流通市值 |
| 技术   | rsi_14, bias_20, vwap_dev, vol_zscore     | RSI、乖离率、VWAP偏离、波动率Z-score   |

### 6.2 预处理管线

```
原始因子 → MAD去极值(5×MAD) → 行业dummy+log市值OLS残差 → Z-score → clip(-5,5)
```

中性化只用于20个量价因子，**不用于权重通道的3个因子**。

---

## 7. 已知局限与后续方向

### 7.1 当前局限

1. **T+1 隐式但未显式建模**：回测引擎的 T+1 通过"上日 score → 当日 pct_chg 计收益"实现，但模型本身不预测 t+1 到 t+5 的路径，只预测端点
2. **无宏观/政策信号**：12维市场特征仅包含量价聚合，不含利率、汇率、政策文本等宏观信息
3. **无风控集成**：当前模拟交易未接入已编写的 risk_manager（止损/周亏损/回撤熔断），实际操作依赖人工判断
4. **IC 长期衰减**：2024 年 RankIC ~0.06，2026 Q1 ~0.045，Q2 出现反转。模型在风格剧烈切换时（如 5/30 科创50 -5%）适应慢
5. **行业轮动未被充分利用**：v4 的行业轮动因子有正向贡献但未被 v1 继承

### 7.2 后续方向

- **Sharpe-aware loss 合并入 v1**（v7 已验证有效，夏普 +41%）
- **动态 n 和 k**：市场低波动时扩大持仓数，高波动时集中（v4 已探索 n=4~14）
- **日内信号**：当前仅用日频，分钟级数据可能捕获更精细的 alpha
- **风控自动化**：将 stop-loss 和 max_drawdown 熔断集成到 daily_ops 流程中

---

## 附录：模型文件对照

| 模型         | 训练脚本                       | Checkpoint                  | 推理入口                         |
| ------------ | ------------------------------ | --------------------------- | -------------------------------- |
| LGBM         | `code/models/lgbm_baseline.py` | `lgbm.pkl`                  | `daily_signal --model lgbm`      |
| MLP          | `code/models/mlp_baseline.py`  | `mlp.pt`                    | (无独立推理)                     |
| GRU-Att      | `code/models/gru_att.py`       | `gru_att.pt`                | `daily_signal --model gru`       |
| MASTER v1    | `code/models/master.py`        | `master.pt`                 | `daily_signal --model master`    |
| MASTER v2    | `code/models/master_v2.py`     | `master_v2.pt`              | `daily_signal --model master_v2` |
| MASTER v3    | `code/models/master_v3.py`     | `master_seed{1337,2024}.pt` | `daily_signal --model master_v3` |
| MASTER v4-v7 | `code/models/master_v{N}.py`   | `v{N}/checkpoints/`         | (搜索实验)                       |
| Ensemble     | `code/models/ensemble.py`      | —                           | `daily_signal --model ensemble`  |

---

_报告生成日期：2026-06-03 | 基于项目 commit c038711 及后续修改_
