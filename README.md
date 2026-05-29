# A股趋势预测与模拟交易 (深度学习基础大作业 2026)

基于深度学习的 A 股短期趋势预测与模拟交易系统。
预测对象: 未来 5 日横截面 rank-percentile。
主模型: **MASTER (AAAI 2024)** — Market-guided Stock Transformer，含时序与横截面双轴注意力。

> 数据字段说明见 `DATA_README.md` (科大云盘 README, 已转为附录)。
> 详细进度与产出物见 `PROGRESS.md`。完整实验报告见 `output/reports/report.html`。

## 环境与安装

```bash
# 创建 conda 环境
conda create -n astock python=3.10
conda activate astock

# 安装 PyTorch (RTX 50 系必须 cu128, 其他显卡按官网选择对应版本)
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 安装其余依赖
pip install -r requirements.txt
```

测试: `python -m code.check_env`。

## 数据准备

数据存放在科大云盘 (USTC PAN), 群组"深度学习基础-2026"。下载到项目根目录后, 结构如下:

```
A股数据/
├── basic.csv                 # 全市场股票基础信息
├── trade_cal.csv             # 交易日历
├── daily/                    # 日频量价 (每个交易日一个 csv)
├── market/                   # 三大指数日频
├── index_weight/             # 沪深300月度权重
├── metric/                   # 基本面指标 (可选, 已用)
├── moneyflow/                # 资金流向 (可选, 已用)
├── news/                     # 新闻快讯 (可选, 未用)
└── stock_st/                 # 每日ST股票列表
```

字段含义见 `DATA_README.md`。

## 完整复现流程

```bash
conda activate astock

# 1. 数据预处理 (Day 1)
python -m code.data.build_panel       # 整合 daily/metric/moneyflow → panel.parquet (~10 min)
python -m code.data.universe          # 中证800近似池 (每月 circ_mv 前 800)
python -m code.data.validate          # 6 项数据正确性检查

# 2. 特征工程 (Day 2)
python -m code.features.factors          # 20 个原始因子
python -m code.features.labels           # 5 日 forward log return + 横截面 rank
python -m code.features.neutralize       # MAD + 行业/市值中性化 + Z-score
python -m code.features.market_features  # 12 维市场状态信号 (train 段标准化)

# 3. 模型训练
python -m code.models.lgbm_baseline      # LightGBM (~8s, 83 trees)
python -m code.models.mlp_baseline       # MLP 基线 (作业建议)
python -m code.models.gru_att            # GRU+Att (~17 min)
python -m code.models.master             # MASTER v1 主模型 (~5 min)
python -m code.models.master_v2          # 加深加宽实验 (反例)
python -m code.models.master_v3          # multi-seed 平均 (~11 min)

# 4. 回测 (真实约束: T+1 + 涨跌停 + 双边手续费)
python -m code.backtest.engine --model master --n 10 --k 2          # 真实约束 (默认)
python -m code.backtest.engine --model master --n 10 --k 2 --no-fee --no-limit  # 理想化对比

# 5. Ensemble + IC 分段分析 + 报告
python -m code.models.ensemble
python -m code.report.ic_analysis        # 制度断点检测
python -m code.report.loss_curves        # 训练 loss 曲线
python -m code.report.build_report --include-v2   # 生成 HTML 报告
```

## 模拟交易 (6/1 起每日盘后)

初始资金: **1,000,000 元 (100万)**

```bash
# 首日 (建仓): 选 top-10 等权 + 计算仓位
python -m code.live.daily_signal --date 20260601 --model master --n 10 --k 2
python -m code.live.position_size --signal output/signals/20260601_master.csv --capital 1000000

# 第二天起 (换仓 k=2): 用 --portfolio 传入当前持仓
python -m code.live.daily_signal --date 20260602 --model master --n 10 --k 2 \
    --portfolio "000001.SZ,600000.SH,..."
```

输出 `output/signals/{date}_master.csv`, 按 score 排序的 (action, ts_code, name, score) 清单。

## 项目结构

```
code/
├── data/             # 数据预处理
├── features/         # 特征工程 (因子/标签/中性化/市场特征)
├── models/           # lgbm / mlp / gru / master(v1+v2+v3) / ensemble
├── backtest/         # 日频回测引擎 (T+1+涨跌停+手续费+ST)
├── live/             # 模拟交易当日信号生成
├── report/           # HTML 报告 / IC 分段分析 / loss 曲线
└── check_env.py

cache/                # 中间产物 (panel.parquet 等)
output/
├── checkpoints/      # 模型权重
├── signals/          # 模型预测 + 模拟交易清单
└── reports/          # HTML 报告 + 8+ 张诊断图
```

## 主要结果 (2024-20260528 OOS 574 交易日, 真实约束, 2026-05-29 重训)

| 模型                       | 类型     | RankIC     | 年化 ICIR | 年化收益  | 夏普     | 最大回撤   |
| -------------------------- | -------- | ---------- | --------- | --------- | -------- | ---------- |
| HS300 基准                 | -        | -          | -         | 19.1%     | 1.06     | -15.7%     |
| LightGBM                   | 树       | 0.0454     | 5.27      | 24.2%     | 1.13     | -22.5%     |
| MLP                        | DL 基线  | 0.0467     | 5.64      | 30.6%     | 1.20     | -20.7%     |
| GRU+Att                    | 时序 DL  | 0.0427     | 4.29      | 18.9%     | 0.71     | -31.6%     |
| MASTER v1                  | 主模型   | 0.0574     | 6.14      | 40.5%     | 2.00     | **-11.1%** |
| **MASTER v3 (multi-seed)** | **SOTA** | **0.0596** | **6.83**  | **52.5%** | **2.22** | -17.3%     |
| Ensemble                   | 加权     | 0.0550     | 5.54      | 29.9%     | 1.67     | -13.2%     |

**主模型选 MASTER v3**: SOTA 表现 (夏普 2.22, 总收益 106.9%), multi-seed 平均在数据分布变化时更稳健。
保守备选 MASTER v1 (最低回撤 -11.1%).

## 关键设计决策

| 决策           | 选择                                                  | 理由                                                  |
| -------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| 股票池         | 中证 800 近似 (月调)                                  | 流动性最优, 与 FDL2026 比赛交易范围对齐, 计算资源可控 |
| 预测对象       | 5 日 forward log return 的横截面 rank                 | rank 横截面稳定, 5 日比 1 日噪声小                    |
| 中性化         | MAD 去极值 + 行业 dummy + log 市值 OLS 残差 + Z-score | 消除行业和市值聚类, 仅留 alpha                        |
| 训练/验证/测试 | 2016-2022 / 2023 / 2024-20260528                      | 严格按时间, OOS 574 天足够稳健                        |
| 损失函数       | IC loss + Top-K margin (alpha=0.6)                    | 排序任务匹配 IC, margin 强化 top-K 分离               |
| 持仓策略       | n=10 等权, k=2 换手                                   | 讲义推荐起点, 平衡集中度和换手成本                    |
| 标准化         | market_features 用 train 段 mean/std (修复过的泄露)   | 严格遵守"不使用未来信息"铁律                          |

## 已知局限 (诚实声明)

详见 HTML 报告"7. 局限与诚实声明"章节。核心:

- 未做财报公告日错位 (基本面 metric 直接用了报告期日期, 未滞后 30-60 天)
- 未模拟订单滑点和成交量约束
- 单次切分训练, 未做 walk-forward; **2026Q2 RankIC=-0.094 提示模型对最近市场失去预测力**
- 新闻情绪因子未实现 (news/ 数据存在)
- 股票池为中证800近似 (~800 只), 非全A股 5000 只 (流动性/计算量考虑)

## 数据泄露自查清单

- [x] 因子按日横截面中性化 (无未来均值/方差)
- [x] market_features 用 train 段(<=2022) mean/std 标准化
- [x] 数据集严格按时间切分, 无随机打乱
- [x] universe 每月用当月可见 circ_mv 重新选 (无回顾性偏差)
- [x] 回测中 ST 用当日 stock_st 名单动态剔除
- [x] portfolio 来自上一日 score → 当日 pct_chg 计收益 (隐式 T+1)
- [x] 标签时间 (t+1 ~ t+5) 严格晚于特征时间 (≤ t)

## 组员与分工

| 姓名 | 学号 | 主要分工 |

> | ---- | ---- | -------- |
> | 阙宇涵 | PB24261891 | 模型框架设计、代码编写运行与审查、实验内容设计与比较、模拟A股操盘 |
> | 卞昌坤 | PB24261894 | 代码运行与审查、模拟A股操盘 |
> | 徐郑潇 | PB24261893 | 代码运行与审查、模拟A股操盘 |

## 提交校验

- [x] requirements.txt
- [x] README.md
- [x] 完整源代码
- [x] HTML 实验报告 (output/reports/report.html)
- [ ] 组员姓名/学号/分工 (报告与 README)
- [ ] 模拟交易截图 (6/12 后补)
- [ ] 模拟交易经验总结 (6/12 后补)

---

# 附录: 数据字段说明

详见 `DATA_README.md`。
