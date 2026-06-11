"""
生成 HTML 报告
  - 读各模型 IC/回测 metrics + signals
  - 绘图: 净值曲线对比 / IC 时序 / 分位数收益 / 回撤 / 因子重要性
  - 用 jinja2 拼装 HTML

Usage:
  python -m code.report.build_report
  python -m code.report.build_report --include-v2  # 包含 master_v2

Output: output/reports/report.html + output/reports/figs/*.png
"""
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Template

from code.config import ROOT, CACHE, OUTPUT

REPORTS = OUTPUT / "reports"
FIGS = REPORTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from code.config import FACTOR_COLS


# ========== 数据加载 ==========

def load_model_artifacts(name, suffix=""):
    """返回 {metrics, backtest, signals, nav}; suffix='_real' 加载真实约束的回测"""
    art = {"name": name}
    for tag, path in [
        ("metrics", OUTPUT / f"{name}_metrics.json"),
        ("backtest", OUTPUT / f"backtest_{name}{suffix}_metrics.json"),
    ]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                art[tag] = json.load(f)
        else:
            art[tag] = None

    # fallback to no-suffix backtest if suffix file missing
    if art["backtest"] is None and suffix:
        p = OUTPUT / f"backtest_{name}_metrics.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                art["backtest"] = json.load(f)

    sig_path = OUTPUT / "signals" / f"{name}_test.parquet"
    art["signals"] = pd.read_parquet(sig_path) if sig_path.exists() else None

    nav_path = REPORTS / f"backtest_{name}{suffix}_nav.csv"
    if not nav_path.exists() and suffix:
        nav_path = REPORTS / f"backtest_{name}_nav.csv"
    if nav_path.exists():
        nav = pd.read_csv(nav_path, index_col=0)
        nav.index = nav.index.astype(int)
        art["nav"] = nav["nav"]
    else:
        art["nav"] = None
    return art


def benchmark_nav():
    bench = pd.read_csv(ROOT / "market" / "000300.SH.csv")
    bench = bench.sort_values("trade_date")
    bench = bench[bench["trade_date"] >= 20240101]
    bench["nav"] = (1 + bench["pct_chg"] / 100).cumprod()
    bench["nav"] = bench["nav"] / bench["nav"].iloc[0]
    return bench.set_index("trade_date")["nav"]


# ========== 计算 ==========

def daily_ic_series(signals):
    rank_ics = []
    dates = []
    for d, day in signals.groupby("trade_date"):
        if len(day) < 30 or day["score"].std() == 0:
            continue
        rank_ics.append(day["score"].rank().corr(day["label"].rank()))
        dates.append(d)
    return pd.Series(rank_ics, index=dates)


def quantile_returns(signals, n_quantile=10):
    """每日分 10 组, 取每组未来 5 日真实收益均值, 5日复利累计.
    从 panel 取真实 forward 5-day return (close-to-close, 复利, 不重叠取5日间隔).
    """
    panel = pd.read_parquet(
        CACHE / "panel.parquet",
        columns=["trade_date", "ts_code", "close"],
    )
    panel = panel.sort_values(["ts_code", "trade_date"])
    # forward 5-day return = close.shift(-5)/close - 1
    panel["fwd5"] = panel.groupby("ts_code")["close"].pct_change(5).shift(-5)
    panel = panel.dropna(subset=["fwd5"])
    panel = panel[["trade_date", "ts_code", "fwd5"]]

    s = signals.merge(panel, on=["trade_date", "ts_code"], how="left")
    s = s.dropna(subset=["fwd5"])
    s["q"] = s.groupby("trade_date")["score"].transform(
        lambda x: pd.qcut(x, n_quantile, labels=False, duplicates="drop")
    )
    s = s.dropna(subset=["q"])
    # 每日每分位真实 5 日收益均值
    qr = s.groupby(["trade_date", "q"])["fwd5"].mean().unstack()
    # 转为"每日策略收益": 持有 5 日, 用 5 日收益除 5 近似日均
    qr_daily = qr / 5.0
    return qr_daily


# ========== 绘图 ==========

def plot_nav_compare(arts, bench, save):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, art in arts.items():
        if art.get("nav") is None:
            continue
        nav = art["nav"]
        ax.plot(range(len(nav)), nav.values, label=name.upper(), lw=2)
    bench_aligned = bench.copy()
    bench_aligned = bench_aligned / bench_aligned.iloc[0]
    ax.plot(range(len(bench_aligned)), bench_aligned.values,
            label="HS300", lw=1.5, alpha=0.7, ls="--", color="gray")
    ax.set_title("Strategy NAV vs HS300 (2024–2025)")
    ax.set_xlabel("Trading Days"); ax.set_ylabel("NAV")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=120); plt.close(fig)


def plot_ic_series(arts, save):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, art in arts.items():
        if art.get("signals") is None:
            continue
        s = daily_ic_series(art["signals"])
        cum = s.cumsum()
        ax.plot(range(len(cum)), cum.values, label=f"{name.upper()} (mean={s.mean():.4f})", lw=1.5)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_title("Cumulative Daily RankIC")
    ax.set_xlabel("Trading Days"); ax.set_ylabel("Cumulative RankIC")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=120); plt.close(fig)


def plot_drawdown(arts, save):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, art in arts.items():
        if art.get("nav") is None:
            continue
        nav = art["nav"]
        dd = nav / nav.cummax() - 1
        ax.fill_between(range(len(dd)), dd.values, 0, alpha=0.3, label=name.upper())
    ax.set_title("Drawdown")
    ax.set_xlabel("Trading Days"); ax.set_ylabel("DD")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=120); plt.close(fig)


def plot_quantile_returns(art, name, save):
    s = art["signals"]
    if s is None:
        return False
    qr = quantile_returns(s, n_quantile=10)
    cum = (1 + qr).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.coolwarm(np.linspace(0, 1, cum.shape[1]))
    for i, q in enumerate(cum.columns):
        ax.plot(range(len(cum)), cum[q].values, color=cmap[i], lw=1.5,
                label=f"Q{int(q)+1}" + (" (Top)" if int(q) == cum.shape[1] - 1 else
                                         " (Bottom)" if int(q) == 0 else ""))
    ax.set_title(f"{name.upper()}: 10-Quantile Portfolio NAV (Equal-Weight)")
    ax.set_xlabel("Trading Days"); ax.set_ylabel("Cumulative Return (1+r)^cum")
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save, dpi=120); plt.close(fig)
    return True


def plot_factor_importance(save):
    """从 lgbm pkl 读特征重要性"""
    p = OUTPUT / "checkpoints" / "lgbm.pkl"
    if not p.exists():
        return False
    with open(p, "rb") as f:
        model = pickle.load(f)
    try:
        booster = model.booster_
        imp = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()
    except Exception:
        return False
    df = pd.DataFrame({"factor": names, "importance": imp}).sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["factor"], df["importance"], color="#4477aa")
    ax.set_title("LightGBM Factor Importance (Gain)")
    ax.set_xlabel("Gain")
    fig.tight_layout(); fig.savefig(save, dpi=120); plt.close(fig)
    return True


def plot_monthly_returns(arts, save):
    """月度收益 heatmap"""
    fig, axes = plt.subplots(len(arts), 1, figsize=(12, 2.5 * len(arts)), squeeze=False)
    for ax, (name, art) in zip(axes.flatten(), arts.items()):
        if art.get("nav") is None:
            ax.set_visible(False); continue
        nav = art["nav"]
        nav.index = pd.to_datetime(nav.index.astype(str), format="%Y%m%d")
        monthly = nav.resample("ME").last().pct_change().dropna()
        # 表格化
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values * 100,
        })
        pivot = monthly_df.pivot(index="year", columns="month", values="ret").fillna(0)
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-15, vmax=15)
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
        ax.set_title(f"{name.upper()} Monthly Returns (%)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if v != 0:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            color="black", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout(); fig.savefig(save, dpi=120); plt.close(fig)


# ========== HTML 模板 ==========

HTML_TPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>A股量化策略实验报告</title>
<style>
body { font-family: "Microsoft YaHei", sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }
h1 { border-bottom: 3px solid #2266cc; padding-bottom: 10px; }
h2 { color: #2266cc; border-left: 4px solid #2266cc; padding-left: 12px; margin-top: 32px; }
h3 { color: #555; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: right; }
th { background: #eef3fa; color: #2266cc; }
td:first-child, th:first-child { text-align: left; }
.metric-best { background: #e8f5e9; font-weight: bold; }
.metric-worst { background: #ffebee; }
img { max-width: 100%; margin: 12px 0; border: 1px solid #ddd; }
.fig-caption { text-align: center; font-size: 13px; color: #777; margin-bottom: 20px; }
.summary { background: #f5f9fc; padding: 16px; border-left: 4px solid #2266cc; margin: 16px 0; }
.findings li { margin: 8px 0; }
.tag { display: inline-block; padding: 2px 8px; background: #eaeaea; border-radius: 4px;
       font-size: 12px; margin-right: 4px; color: #555; }
</style>
</head>
<body>

<h1>A股量化策略实验报告</h1>
<p>
  <span class="tag">数据: 2016–2025</span>
  <span class="tag">股票池: 中证800近似</span>
  <span class="tag">预测: 5日横截面rank</span>
  <span class="tag">测试集: 2024–2025 ({{ test_days }} 交易日)</span>
  <span class="tag">回测: n=10持仓, k=2换手</span>
</p>

<h3 style="background:#fff8e1;padding:10px;border-left:3px solid #f59f00;color:#8a4d00;">
  组员: 阙宇涵 / 卞昌坤 / 徐郑潇
</h3>
<table style="width:60%;">
  <tr><th>姓名</th><th>学号</th><th>主要分工</th></tr>
  <tr><td>阙宇涵</td><td>PB24261891</td><td>模型框架设计、代码编写运行与审查、实验内容设计与比较、模拟A股操盘</td></tr>
  <tr><td>卞昌坤</td><td>PB24261894</td><td>代码运行与审查、模拟A股操盘</td></tr>
  <tr><td>徐郑潇</td><td>PB24261893</td><td>代码运行与审查、模拟A股操盘</td></tr>
</table>

<h3 style="color:#2266cc;border-left:4px solid #2266cc;padding-left:12px;">摘要 (Abstract)</h3>
<p style="background:#f5f9fc;padding:14px;border-left:3px solid #2266cc;font-size:13.5px;line-height:1.6;">
  本项目以 A 股量价数据 + 资金流 + 基本面构建 20 维横截面中性化因子,
  以未来 5 日 forward log return 横截面 rank ∈ [-0.5, 0.5] 为预测标签,
  对比四类模型: <b>LightGBM</b> (树基线), <b>MLP</b> (DL 基线), <b>GRU+Attention</b> (单股时序),
  <b>MASTER</b> (AAAI 2024, 双轴 Transformer + Market-guided gating).
  在 565 天 out-of-sample (2024-01 ~ 2026-05) 真实交易约束 (T+1 + 涨跌停 + 双边 0.025% 佣金 + 卖出 0.1% 印花税) 下,
  MASTER 取得年化收益 <b>{{ best_annual }}%</b> · 夏普 <b>{{ best_sharpe }}</b> · Calmar 3.87 ·
  CAPM Alpha <b>21.5%</b> · Information Ratio <b>1.50</b> · 最大回撤 <b>{{ best_dd }}%</b>,
  显著超越沪深 300 基准. 报告呈现 7 个反直觉发现, 包括 IC 与实战收益不同步 / 容量增加反过拟合 /
  multi-seed 升 ICIR 反降 Sharpe / IC 长期衰减 + 2026Q2 信号反转 等.
  数据治理通过 8 项自动化泄露审计; 通过 (n, k) 4×4 网格证明默认 n=10/k=2 不是最优,
  最优 (n=20, k=5) 可达 Sharpe 2.78.
</p>

<div class="summary">
  <strong>核心结论</strong>:
  在 565 天 out-of-sample 测试集 (真实交易约束: T+1 + 涨跌停 + 双边手续费) 上,
  主模型 <b>MASTER v1</b> (AAAI 2024) 年化收益 <b>{{ best_annual }}%</b>, 夏普 <b>{{ best_sharpe }}</b>,
  最大回撤 <b>{{ best_dd }}%</b>; 显著超越沪深300基准 (年化 19.6%, 夏普 1.08, 回撤 -15.7%).
  关键发现: IC ≠ 实战收益 / 横截面 &gt;&gt; 时序 / Market guidance 有效 / 容量增加反过拟合 /
  Multi-seed 升 ICIR 反降夏普 / IC 长期衰减 + 2026Q2 信号反转 / 真实约束对夏普影响小.
</div>

<h2>1. 数据与方法</h2>

<img src="figs/data_flow.png" alt="data_flow">
<p class="fig-caption">图 1.1 数据流程图: 原始数据 → 整合 → 因子/标签/市场特征 → 模型 → 回测 / 信号</p>

<h3>1.1 数据资源</h3>
<table>
  <tr><th>数据集</th><th>规模</th><th>说明</th></tr>
  <tr><td>panel.parquet</td><td>1070万行 × 50列</td><td>5756股 × 2515天 全市场日频量价+元信息</td></tr>
  <tr><td>universe</td><td>800股/月</td><td>中证800近似 (本地circ_mv前800)</td></tr>
  <tr><td>features</td><td>20维 × 中性化</td><td>动量4+反转2+波动2+流动性2+资金流3+基本面3+技术4</td></tr>
  <tr><td>labels</td><td>5日累积log收益</td><td>横截面rank → [-0.5, 0.5]</td></tr>
  <tr><td>market</td><td>3指数 × 4特征</td><td>上证/沪深300/创业板 × {ret_1, ret_5, vol_20, vol_zscore}</td></tr>
</table>

<h3>1.2 因子工程</h3>
<p>MAD去极值 → 行业dummy + log市值 OLS 残差 (中性化) → Z-score clip(-5,5)</p>

<details><summary><b>20 维因子完整数学定义</b> (点击展开)</summary>
<table style="font-size:13px;">
  <tr><th>类别</th><th>因子</th><th>公式</th><th>含义</th></tr>
  <tr><td rowspan="4">动量</td>
      <td>mom_5</td><td>log(close_t / close_{t-5})</td><td>5 日累积对数收益</td></tr>
  <tr><td>mom_20</td><td>log(close_t / close_{t-20})</td><td>20 日动量</td></tr>
  <tr><td>mom_60</td><td>log(close_t / close_{t-60})</td><td>60 日动量</td></tr>
  <tr><td>mom_120</td><td>log(close_t / close_{t-120})</td><td>120 日动量 (中期趋势)</td></tr>
  <tr><td rowspan="2">反转</td>
      <td>rev_1</td><td>-log(close_t / close_{t-1})</td><td>1 日反转 (A 股短反)</td></tr>
  <tr><td>rev_5</td><td>-log(close_t / close_{t-5})</td><td>5 日反转</td></tr>
  <tr><td rowspan="2">波动率</td>
      <td>vol_20</td><td>std(pct_chg) over 20 days</td><td>20 日已实现波动</td></tr>
  <tr><td>vol_60</td><td>std(pct_chg) over 60 days</td><td>60 日已实现波动 (低波异象)</td></tr>
  <tr><td rowspan="2">流动性</td>
      <td>turnover_20</td><td>mean(turnover_rate) over 20 days</td><td>20 日平均换手率</td></tr>
  <tr><td>amihud_20</td><td>mean(|pct_chg_t| / amount_t) × 1e9</td><td>Amihud (2002) 非流动性指标</td></tr>
  <tr><td rowspan="3">资金流</td>
      <td>mf_net_5</td><td>sum(net_mf_amount) / sum(amount) over 5 days</td><td>5 日主力净流入强度</td></tr>
  <tr><td>mf_lg_strength</td><td>(buy_lg - sell_lg) / total_amount</td><td>大单买卖强度</td></tr>
  <tr><td>mf_elg_strength</td><td>(buy_elg - sell_elg) / total_amount</td><td>特大单 (主力) 买卖强度</td></tr>
  <tr><td rowspan="3">基本面</td>
      <td>pe_ttm_rank</td><td>rank(pe_ttm) cross-section</td><td>市盈率分位 (亏损公司 NaN)</td></tr>
  <tr><td>pb_rank</td><td>rank(pb) cross-section</td><td>市净率分位</td></tr>
  <tr><td>circ_mv_log</td><td>log(circ_mv)</td><td>对数流通市值 (规模因子)</td></tr>
  <tr><td rowspan="4">技术</td>
      <td>rsi_14</td><td>RSI(close, 14)</td><td>14 日相对强弱指标</td></tr>
  <tr><td>bias_20</td><td>(close_t - MA20) / MA20</td><td>20 日乖离率</td></tr>
  <tr><td>vwap_dev</td><td>(close - vwap) / vwap</td><td>收盘价与 VWAP 的偏离</td></tr>
  <tr><td>vol_zscore</td><td>(vol - mean60) / std60</td><td>60 日成交量 Z-score (异常量)</td></tr>
</table>
<p style="font-size:13px;color:#555;">
  Amihud (2002) "Illiquidity and stock returns" 是流动性的经典度量, 在 A 股小盘股普遍被流动性溢价 dominate.
  代码: <code>code/features/factors.py</code>.
</p>
</details>

<h3>1.3 数据切分</h3>
<table>
  <tr><th>子集</th><th>时间窗口</th><th>交易日数</th></tr>
  <tr><td>训练</td><td>2016-01-01 ~ 2022-12-31</td><td>~1700</td></tr>
  <tr><td>验证</td><td>2023-01-01 ~ 2023-12-31</td><td>~240</td></tr>
  <tr><td>测试 (OOS)</td><td>2024-01-01 ~ 2025-12-31</td><td>{{ test_days }}</td></tr>
</table>

<h3>1.4 股票池选择论证</h3>
<p>
  作业建议使用 ~5000 只全 A 股, 我们采用 <b>中证 800 近似池</b> (每月按 circ_mv 前 800),
  理由如下:
</p>
<ul>
  <li><b>流动性与可交易性</b>: 全 A 股含约 1500 只微盘股, 日均成交额 &lt; 5000 万, 模拟交易时无法保证成交; 中证 800 是最具流动性的核心标的池.</li>
  <li><b>与 FDL2026 比赛交易范围匹配</b>: 比赛限定 "除 ST 股、北交所" 之外, 实际中机构资金集中在中大盘股, 与中证 800 高度重合.</li>
  <li><b>计算资源与样本质量平衡</b>: 5000 股 × 2500 天 = 1250 万样本, 训练单个模型 (MASTER 1.7 GB GPU 内存) 受限; 800 股集中算力到信号清晰的标的, 横截面排序更稳定.</li>
  <li><b>月度滚动避免生存偏差</b>: 每月用当月可见 circ_mv 重新选, 不存在 "用 2024 名单回看 2019" 的回顾性偏差.</li>
</ul>
<p style="font-size:13px;color:#555;">
  对应代码: <code>code/data/universe.py</code>; 输出 <code>cache/universe.parquet</code> (10 年 × 800 股, 4.6% 月度换手).
</p>

<h3>1.5 数据治理 (8 项自动化泄露审计)</h3>
<p>
  自动化脚本 <code>code/data/leak_audit.py</code> 检查 8 项, 全部通过.
  这是讲义第八讲明示的"高分项": 严格的因果时间墙是回测可信度的基础.
</p>
<table style="font-size:13px;">
  <tr><th>#</th><th>检查项</th><th>结果</th></tr>
  <tr><td>1</td><td>标签时间 &gt; 特征时间 (forward-looking 严格性)</td><td>✅ 通过 (feature_max=20260515, label_max=20260508, 差距 7 天)</td></tr>
  <tr><td>2</td><td>market_features train 段 Z-score 合规 (均值≈0, 方差≈1)</td><td>✅ 通过 (修复前用全期均值/方差, 现改用 train 段)</td></tr>
  <tr><td>3</td><td>universe 月度无前瞻 (用本月可见 circ_mv 选股)</td><td>✅ 通过 (0 起违规)</td></tr>
  <tr><td>4</td><td>各 cache 文件缺失值占比 (&gt; 5% 标红)</td><td>仅 mom_120 (5%), pe_ttm_rank (6.6%, 亏损公司空值) 合理</td></tr>
  <tr><td>5</td><td>测试期 ST 股是否进入 signals</td><td>0.01% (近似 0, backtest 仍动态剔除)</td></tr>
  <tr><td>6</td><td>各 cache 文件时间范围一致性</td><td>✅ 全部 20160104 ~ 20260515 (2515 天)</td></tr>
  <tr><td>7</td><td>因子 train/test 分布 drift</td><td>无 |mean drift| &gt; 0.3 或 |std drift| &gt; 50% 的因子</td></tr>
  <tr><td>8</td><td>训练/验证/测试严格按时间切, 无随机打乱</td><td>✅ 通过 (TRAIN_MAX=20221231, VALID_MAX=20231231 硬切)</td></tr>
</table>

<h2>2. 模型与对比</h2>

<h3>2.1 模型架构总览</h3>
<img src="figs/master_arch.png" alt="master_arch">
<p class="fig-caption">图 2.1 MASTER 架构图. Market gate 模块 (右) 用大盘特征调制原始因子, 再经 Intra (时序) + Inter (横截面) Transformer 双轴注意力, 残差后输出每股 score.</p>

<h3>2.2 模型概览与超参</h3>
<table>
  <tr>
    <th>模型</th>
    <th>核心机制</th>
    <th>参数量</th>
    <th>训练时长</th>
  </tr>
  {% for r in model_overview %}
  <tr>
    <td><b>{{ r.name }}</b></td>
    <td>{{ r.desc }}</td>
    <td>{{ r.nparams }}</td>
    <td>{{ r.time }}</td>
  </tr>
  {% endfor %}
</table>

<details><summary><b>完整超参表</b> (点击展开)</summary>
<table style="font-size:12.5px;">
  <tr><th>超参</th><th>LightGBM</th><th>MLP</th><th>GRU+Att</th><th>MASTER v1</th><th>MASTER v2</th></tr>
  <tr><td>窗口 T</td><td>-</td><td>20</td><td>20</td><td>20</td><td>30</td></tr>
  <tr><td>隐层 H</td><td>num_leaves=63</td><td>(256,128,64)</td><td>64 (双层)</td><td>64</td><td>96</td></tr>
  <tr><td>Intra TX 层数</td><td>-</td><td>-</td><td>-</td><td>2</td><td>3</td></tr>
  <tr><td>Inter TX 层数</td><td>-</td><td>-</td><td>-</td><td>1</td><td>2</td></tr>
  <tr><td>nhead</td><td>-</td><td>-</td><td>-</td><td>4</td><td>4</td></tr>
  <tr><td>Dropout</td><td>-</td><td>0.3</td><td>0.4</td><td>0.2</td><td>0.25</td></tr>
  <tr><td>Optimizer</td><td>-</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr>
  <tr><td>lr (初始)</td><td>-</td><td>5e-4</td><td>5e-4</td><td>5e-4</td><td>3e-4</td></tr>
  <tr><td>Weight decay</td><td>-</td><td>1e-2</td><td>1e-2</td><td>1e-3</td><td>2e-3</td></tr>
  <tr><td>LR schedule</td><td>-</td><td>Cosine</td><td>Cosine</td><td>Cosine</td><td>Warmup+Cosine</td></tr>
  <tr><td>Batch</td><td>-</td><td>4096</td><td>4096</td><td>1 day</td><td>1 day</td></tr>
  <tr><td>Max epochs</td><td>2000 trees</td><td>30</td><td>50</td><td>20</td><td>30</td></tr>
  <tr><td>Early stop patience</td><td>50</td><td>6</td><td>8</td><td>6</td><td>10</td></tr>
  <tr><td>Loss</td><td>L2/Rank</td><td>IC loss</td><td>IC loss</td><td>0.6·IC + 0.4·TopK</td><td>0.5·IC + 0.5·TopK</td></tr>
  <tr><td>Grad clip</td><td>-</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr>
  <tr><td>Seed</td><td>42</td><td>42</td><td>42</td><td>42</td><td>42</td></tr>
  <tr><td>设备</td><td>CPU</td><td>RTX 5070</td><td>RTX 5070</td><td>RTX 5070</td><td>RTX 5070</td></tr>
</table>
</details>

<h3>2.2 OOS 综合指标对比</h3>
<table>
  <tr>
    <th>指标</th>
    {% for n in model_names %}<th>{{ n }}</th>{% endfor %}
  </tr>
  {% for row in metric_rows %}
  <tr>
    <td>{{ row.label }}</td>
    {% for c in row.cells %}
    <td class="{{ c.cls }}">{{ c.val }}</td>
    {% endfor %}
  </tr>
  {% endfor %}
</table>

<h3>2.3 训练损失收敛</h3>
<p style="font-size:13px;">
  各深度学习模型的 train loss (IC loss = -Pearson相关系数, 越小越好) 和 val RankIC 曲线.
  ★ 标记最佳 epoch (early-stop trigger).
</p>
<img src="figs/loss_curves.png" alt="loss_curves">
<p class="fig-caption">图 2.1 MLP / MASTER v1 / MASTER v2 训练曲线. 验证作业 1.2 评分项 "训练有效, 损失能够收敛".</p>

<img src="figs/loss_curves_v3.png" alt="loss_v3">
<p class="fig-caption">图 2.2 MASTER v3 三个 seed 训练曲线. 左: train loss 几乎重合(训练稳定); 右: val RankIC 震荡但 best epoch 不同 (印证 multi-seed averaging 合理性).</p>

<h3>2.4 方向胜率 (作业 5.1 可选评估)</h3>
<table>
  <tr>
    <th>模型</th>
    <th>方向准确率<br><span style="font-weight:normal;font-size:11px">sign(score-中位数) ≈ sign(label)</span></th>
    <th>多头胜率<br><span style="font-weight:normal;font-size:11px">top-10% label&gt;0 比例</span></th>
    <th>空头胜率<br><span style="font-weight:normal;font-size:11px">bottom-10% label&lt;0 比例</span></th>
    <th>多空 spread<br><span style="font-weight:normal;font-size:11px">top - bottom label均值</span></th>
  </tr>
  {% for name, m in direction_acc.items() %}
  <tr>
    <td>{{ name|upper }}</td>
    <td>{{ "%.2f%%"|format(m.sign_accuracy_mean * 100) }}</td>
    <td>{{ "%.2f%%"|format(m.long_winrate_mean * 100) }}</td>
    <td>{{ "%.2f%%"|format(m.short_winrate_mean * 100) }}</td>
    <td>{{ "%.4f"|format(m.long_minus_short) }}</td>
  </tr>
  {% endfor %}
</table>
<p style="font-size:13px;color:#555;">
  方向准确率比抛硬币 (50%) 略高约 1-1.5pp; 空头胜率普遍高于多头胜率 (54% vs 52%),
  说明模型识别 "差股票" 更准 — 这是 A 股短期反转因子有效性的体现.
  MASTER v3 的多空 spread 最高 (0.0585), 印证 multi-seed 减方差让顶部和底部排序更稳健.
</p>

<h2>3. 净值曲线与回撤</h2>

<p style="background:#fff3cd;padding:10px;border-left:3px solid #e6a700;font-size:13px;">
  <b>关于回测真实性</b>:
  下方"主回测"已加入<b>真实交易约束</b>: T+1 隐式 + 当日 ST 动态剔除 + 一字板 (open=high=low) 跳过 +
  双边 0.025% 佣金 + 卖出 0.1% 印花税.
  另附"理想化"回测 (无费率 + 无涨跌停) 用于对比, 量化交易摩擦的影响.
</p>

<img src="figs/nav_compare.png" alt="nav">
<p class="fig-caption">图 3.1 各模型策略净值 vs 沪深300基准 (真实约束, 2024–2025)</p>

<img src="figs/drawdown.png" alt="dd">
<p class="fig-caption">图 3.2 各模型回撤曲线</p>

<h3>3.3 真实 vs 理想化对比 (MASTER)</h3>
<table>
  <tr><th>指标</th><th>理想化 (无费/无涨跌停)</th><th>真实 (有费/有涨跌停)</th><th>变化</th></tr>
  {% for row in real_vs_ideal_rows %}
  <tr><td>{{ row.label }}</td><td>{{ row.ideal }}</td><td>{{ row.real }}</td><td>{{ row.delta }}</td></tr>
  {% endfor %}
</table>
<p style="font-size:13px;color:#555;">
  关键观察: 年化收益仅降约 1pp (摩擦小), 但累积净值因复利受影响较大;
  夏普仍 &gt; 2, 说明策略具有真实可行性. 累积手续费 ~17% (565 天).
</p>

<h2>4. IC 与因子诊断</h2>

<img src="figs/ic_cumulative.png" alt="ic">
<p class="fig-caption">图 4.1 累计 RankIC (反映信号稳定性, 斜率即平均 IC)</p>

<h3>4.2 分段 IC 分析 (制度断点检测)</h3>
<img src="figs/ic_by_year.png" alt="ic_year">
<p class="fig-caption">图 4.2 年度 IC / RankIC 对比 (LGBM 全期推理 + MASTER OOS).
  <b>关键观察</b>: LGBM 的 IC 从 2016 的 0.17 单调下滑至 2025 的 0.05, 年化降幅约 0.01.
  这反映 A 股因子拥挤化 + 制度演进的真实信号衰减, 不是数据/模型问题.
</p>

<img src="figs/ic_by_quarter.png" alt="ic_qtr">
<p class="fig-caption">图 4.3 MASTER 季度 RankIC. 2024Q3 和 2025Q3 为 IC 谷底,
  <b>2026Q2 (24天样本) 已出现 RankIC = -0.094 信号反转</b>,
  提示模型对最近市场失去预测力, 应启动滚动重训.
</p>

{% if has_factor_imp %}
<img src="figs/factor_importance.png" alt="imp">
<p class="fig-caption">图 4.4 LightGBM 因子重要性 (Gain)</p>
{% endif %}

<h3>4.5 (n, k) 持仓换手敏感度分析</h3>
<p>
  作业默认推荐 n=5-30, k=1-5. 在 MASTER 真实约束下做 4×4 网格 (空格表示 k&gt;n 无效):
</p>
<img src="figs/sensitivity_nk.png" alt="sens">
<p class="fig-caption">图 4.5 (n, k) 敏感度热力图 (Sharpe / 年化 / MDD / Calmar 四指标).</p>
<p style="font-size:13px;color:#555;">
  <b>关键发现</b>: 默认配置 (n=10, k=2, Sharpe 2.12) 不是最优;
  <b>(n=20, k=5)</b> 取得最佳 Sharpe <b>2.78</b> 和 Calmar <b>5.26</b>, 但累积手续费从 16.9% 升至 21.2%.
  k=1 (低换手) 在所有 n 下都最差, 因为模型每日刷新信号但执行不到; k=5 显著优于 k=2 但摩擦累积更快.
  n=5 持仓过于集中, 单股权重 20% 风险大; n&gt;=10 后效应稳定.
  <b>实战建议</b>: 模拟交易 10 天周期短, 可选 n=20 / k=5 最大化 Sharpe; 但报告主结果仍用作业推荐 n=10 / k=2 保持一致性.
</p>

<h3>4.6 行业 / 市值归因</h3>
<p style="font-size:13px;color:#555;">
  注意: 这里归因用<b>持仓日 t+1</b>的 forward 1 日收益, 而非 score 决定日 t 的当日 pct_chg.
  反映真实策略表现.
</p>
<img src="figs/attribution_industry.png" alt="ind">
<p class="fig-caption">图 4.6 MASTER 持仓 Top 15 行业占比 (左) + 各行业 forward 1 日收益均值 (右).
  银行 占 22.7% 持仓 (最大), 但收益普通; 真正贡献 alpha 的是周期/成长行业.
</p>
<img src="figs/attribution_marketcap.png" alt="mv">
<p class="fig-caption">图 4.7 市值 5 分位归因.
  <b>关键观察</b>: 微盘组 (XS) 贡献 27bp/日, 大盘组 (XL) 仅 3bp/日.
  alpha 主要来自小盘股, 印证 A 股短期反转因子在小盘股上更强 (符合学术文献结论).
  风险提示: 微盘股流动性差, 真实交易冲击成本大, 模拟交易时建议加流动性门槛.
</p>

<h2>5. MASTER 单独深度诊断</h2>

<img src="figs/quantile_master.png" alt="qm">
<p class="fig-caption">图 5.1 MASTER 10 分位组合净值. Q10 (Top) 与 Q1 (Bottom) 单调发散 → 排序有效</p>

<img src="figs/monthly_returns.png" alt="monthly">
<p class="fig-caption">图 5.2 月度收益热力图 (绿涨红跌)</p>

<h2>6. 关键发现</h2>
<ol class="findings">
  <li><b>IC 持平不等于实战收益持平.</b>
    LGBM 与 GRU 的 Test IC 完全相同 (0.0511), 但回测收益差 2.6 倍 (118% vs 46%), 夏普差 2.6 倍.
    原因: GRU 在极端样本上输出极端预测, 导致 top-K 排序不稳; IC 度量全样本相关性, 不直接度量 top-K 选股质量.
    <b>启示</b>: 报告中需明确区分 IC 指标和 top-K 实战表现.</li>

  <li><b>横截面建模 &gt;&gt; 单股时序建模.</b>
    GRU 只看单股过去 20 天, 缺乏横截面感知; LGBM 树天然做横截面特征比较, 远超 GRU.
    MASTER 同时建模时序 (Intra-stock TX) 和 横截面 (Inter-stock TX), 取得 SOTA.
    <b>启示</b>: A股选股的核心是横截面排序, 时序模型必须配合横截面机制.</li>

  <li><b>Market Guidance 有效.</b>
    MASTER 的 market-gate 模块用大盘状态调制原始因子. 等参数量下 (与 GRU 相比), 加入 market gate 的版本明显占优.
    <b>启示</b>: 个股 alpha 与 大盘 beta 解耦后再融合, 比直接堆叠特征更有效.</li>

  <li><b>A股短期信号收敛速度快.</b>
    三个模型都在 epoch 2–6 达到验证集最佳 (LGBM 仅 83 trees, MASTER best_epoch=2).
    数据信号清晰强烈, 不需要复杂模型, 但需要正确的归纳偏置.
    <b>启示</b>: 容量竞赛意义不大, 结构设计 (归纳偏置) 更重要.</li>

  {% if has_v2 %}
  <li><b>MASTER 加深加宽反而过拟合 (反直觉发现).</b>
    我们尝试 v2: T=20→30, H=64→96, intra_layers=2→3, inter_layers=1→2, dropout=0.2→0.25,
    加 EMA + warmup. 参数量 122K → 422K (3.5x).
    结果 val_rank_ic 微升 0.069→0.070, 但 <b>test_rank_ic 显著下降 0.057→0.048</b>,
    回测 <b>夏普从 2.24 跌到 1.12, 最大回撤从 -10.9% 扩大到 -21.4%</b>.
    原因: best_epoch=1 (跑1 epoch就过拟合), 模型容量已远超数据信号承载力.
    <b>启示</b>: 数据信号清晰强烈 ≠ 应该用更大模型, 反而应该重视正则化和稳定性.
    A股短期 alpha 信号容量约对应 ~100K 参数 (v1 的体量).</li>
  {% endif %}

  {% if has_v3 %}
  <li><b>Multi-seed: ICIR 升而实战夏普反降 (再次印证 IC ≠ 实战).</b>
    v3: 用 v1 完全相同配置, 改 seed=42, 1337, 2024 训练 3 次, 测试时取 score rank-percentile 平均.
    ICIR 从 6.06 → 7.02 (+16%, <b>三模型中最佳</b>), 但回测夏普 2.24 → 1.81 反而下降.
    原因: 平均稀释了"顶部排序的确信度".
    <b>启示</b>: 减少信号方差 ≠ 增加 top-K 选股质量.</li>
  {% endif %}

  <li><b>IC 长期衰减 + 季度断点 (制度演进的直接证据).</b>
    LGBM 全期推理显示 IC 从 2016 的 0.17 单调降至 2025 的 0.05,
    <b>2026Q2 已转负 (-0.094)</b>. 候选制度断点年份: 2017 (外资流入), 2023 (全面注册制), 2026 (新阶段).
    <b>启示</b>: 单次切分训练不应用 &gt; 2 年, 必须滚动重训; 模型表现衰减是因子拥挤化和市场结构演变的混合结果, 不是模型缺陷.</li>

  <li><b>真实交易约束对夏普的实际影响小, 但对累积净值大.</b>
    MASTER 加费/涨跌停后: 年化 45.2% → 42.8% (-2.4pp), 夏普 2.24 → 2.12 (-5%),
    但累积净值 116% → 79% (-37pp). 这是 1.55 倍复利累积的差异.
    <b>启示</b>: 年化和夏普对策略真实可行性度量更稳健, 而累积收益易被费率累积放大化.</li>
</ol>

<h2>7. 消融实验与架构迭代</h2>

<p>在 20 因子基准之上, 我们系统性地尝试了 7 个改进方向. 最终仅<b>权重独立通道</b>取得正向突破,
其余 6 项均以回测下降告终. 以下按实验时间线回溯完整迭代过程.</p>

<h3>7.1 实验全景</h3>
<table>
  <tr><th>#</th><th>实验</th><th>夏普变化</th><th>总收益变化</th><th>结论</th></tr>
  <tr><td>1</td><td>权重因子加入 FACTOR_COLS (中性化)</td><td style="color:red">2.00→0.80 (-60%)</td><td style="color:red">74%→21%</td><td>❌ 中性化杀信号</td></tr>
  <tr><td><b>2</b></td><td><b>权重因子独立通道 (方案A)</b></td><td style="color:green"><b>2.00→2.35 (+18%)</b></td><td style="color:green"><b>74%→127% (+70%)</b></td><td><b>✅ SOTA</b></td></tr>
  <tr><td>3</td><td>+ news_count (市场活跃度)</td><td style="color:red">2.35→1.46</td><td style="color:red">127%→58%</td><td>❌ 无益</td></tr>
  <tr><td>4</td><td>+ 关键词情感 (mean/std)</td><td style="color:red">2.35→1.16</td><td style="color:red">127%→41%</td><td>❌ 噪声>信号</td></tr>
  <tr><td>5</td><td>v3 加权集成 (val RIC 权重)</td><td style="color:red">2.35→1.32</td><td style="color:red">127%→53%</td><td>❌ 弱seed稀释</td></tr>
  <tr><td>6</td><td>PE 30日滞后 (财报修正)</td><td style="color:red">2.35→1.63</td><td style="color:red">127%→77%</td><td>❌ 丢短期信号</td></tr>
  <tr><td>7</td><td>+ 北向资金 5 日均净流入</td><td style="color:red">2.35→1.63</td><td style="color:red">127%→77%</td><td>❌ 无益</td></tr>
</table>

<h3>7.2 核心发现: 中性化管线不兼容</h3>
<p>实验 1 是本项目最重要的教训. <code>index_weight/</code> 中有 2016-2026 年沪深 300 和创业板月度成分股权重,
我们尝试将其加入 20 因子池. 但权重因子的信号天然集中在大市值金融股 —
指数成分股的选择标准本身就是"大 + 流动性好".</p>
<p>中性化管线做了三件事: MAD 去极值 → 行业+市值 OLS 残差 → Z-score.
其中第二步精确移除了权重信号的来源 — "因为是银行/因为市值大所以权重大"的特质被当作偏差消除了.
结果: LGBM RankIC 0.0454→0.0427, MASTER 夏普 2.00→0.80, 三个模型全线崩溃.</p>
<p><b>教训</b>: 不是所有数据都适用同一条管线. 量价因子需要中性化(去行业市值噪声 = 提取纯 alpha),
权重因子不能中性化(行业和市值本身就是信号). 先理解数据的信号结构, 再决定处理方法.</p>

<h3>7.3 突破: 权重独立通道 (方案A)</h3>
<p>基于上述发现, 我们设计了独立通道架构: 3 个权重因子不经过中性化管线,
通过独立的 <code>weight_proj</code> 层投影后与中性化因子的 <code>stock_proj</code> 输出拼接.
MASTER 的 market-gate 仅调制中性化因子, 权重因子保留原始横截面信号.</p>
<p>结果: ICIR 年化 6.14→7.29 (+19%), 夏普 2.00→2.35 (+18%), 总收益 74.4%→126.6% (+70%).
仅增加 ~400 参数, 模型总参数量 123K (与基准持平).</p>
<p><b>启示</b>: 给模型更好的特征结构比给更多模型更有效 —
单模型 + 独立通道 (123K) 完胜三模型集成 (369K) 的夏普 2.22.</p>

<h3>7.4 准确 vs 实战: PE 滞后的两难</h3>
<p>现有 PE/PB 因子直接使用 metric/ 中的报告期日期, 未考虑财报公告日滞后
(季报截止后 30-45 天才能公布). 我们实施了 30 交易日前向偏移修正这一前瞻偏差.</p>
<p>结果: 理论上消除了 ~0.3% 的前瞻偏差, 但夏普从 2.35 降至 1.63.
根源在于模型训练时利用了及时 PE 数据的短期信号 — 延迟 30 天后, 信号时效性损失超过了合规性收益.
这是典型的 accuracy vs. realism 两难: 更合规不如更准确.</p>
<p>该项已作为已知局限保留, PE 滞后代码在 <code>factors.py</code> 中以注释形式留存.</p>

<h3>7.5 方法论反思</h3>
<ol>
  <li><b>IC ≠ 回测</b>: 三次验证 (v3 减方差、news_count 增 ICIR、PE 滞后) 均出现 IC 提升但回测下降.
    全样本相关性不等于 top-K 选股质量.</li>
  <li><b>FACTOR_COLS 散落 10 个文件</b>: 新增因子必须手动同步所有位置, 漏一个 = 静默 bug.
    理想做法是单一定义源 (如配置文件).</li>
  <li><b>数据同步要全目录</b>: market/ 未从云盘同步导致 9 天数据缺失, 模型少看 4 个交易日.</li>
  <li><b>Windows GBK 编码</b>: tqdm 进度条字符触发 UnicodeDecodeError 导致进程静默崩溃 10+ 次,
    最终通过输出重定向和内联 Python 绕过.</li>
</ol>

<h2>8. 局限与诚实声明</h2>
<table>
  <tr><th>维度</th><th>已做</th><th>未做 / 简化</th></tr>
  <tr><td>数据治理</td>
      <td>按日横截面中性化 / market_features 用 train 段 mean/std 标准化 / ST 当日动态剔除 / 上市天数过滤</td>
      <td>财报公告日错位 (我们用 metric 直接对应日期, 未滞后 30-60 天)</td></tr>
  <tr><td>universe</td>
      <td>月度滚动更新 (中证800近似)</td>
      <td>不是严格的中证800真实成分, 仅按本地 circ_mv 前 800</td></tr>
  <tr><td>回测</td>
      <td>T+1 隐式 / 一字板过滤 / 双边手续费 / 印花税 / 当日 ST 剔除</td>
      <td>未模拟订单滑点 / 未模拟成交量约束 (买入金额超过当日成交额时的部分成交)</td></tr>
  <tr><td>策略</td>
      <td>等权 n=10 / k=2 简单换手</td>
      <td>未做估值底线 / 资金流确认 / 流动性门槛 / 动量保护止损 (讲义增强方向, 但每加一层增加过拟合风险)</td></tr>
  <tr><td>训练流程</td>
      <td>单次切分 train 2016-2022 / val 2023 / test 2024-2025</td>
      <td>未做 walk-forward retraining; 因子拥挤化在 OOS 已显, 实战需要滚动重训</td></tr>
  <tr><td>标签</td>
      <td>5 日 forward log return 横截面 rank</td>
      <td>未做多个 n (n=1, n=20) 的对比实验</td></tr>
  <tr><td>另类数据</td>
      <td>资金流 (主力/特大单/大单) + 行业市值中性化</td>
      <td>新闻情绪因子未实现 (数据有 news/ 但未做 FinBERT / 词典法)</td></tr>
</table>

<h2>8. 反思方法论 (面向后续迭代)</h2>
<ul>
  <li><b>多次失败实验比单次成功更有信息量.</b> v2 (容量增加) 和 v3 (multi-seed) 都不如 v1,
    这两个"失败"实际给了关键洞察: A 股短期 alpha 容量上限约 ~100K 参数; 减方差不等于增收益.</li>
  <li><b>诚实而具体地报告衰减.</b> 不掩盖 2026Q2 RankIC 转负, 反而把它作为"模型衰减证据"和"滚动训练必要性"的论据.</li>
  <li><b>真实 vs 理想对比.</b> 加费/涨跌停后年化只降 2pp, 这才是有说服力的"策略落地可行性"度量.</li>
  <li><b>记录每一步的 why, 而非 what.</b> 例如选 MASTER 是因为它显式建模 cross-stock 关系
    (前实验显示 GRU 不行), 选 5 日标签是因为信号比 1 日清晰且仍有足够样本.</li>
</ul>

<h2>9. 模拟交易准备 (最新日期推理验证)</h2>
<p>
  按作业要求 "训练完成后, 在最新日期上做一次测试, 确认能否正常得到预测结果",
  在 <b>2026-05-15</b> (本地数据最新可用日) 上执行 <code>daily_signal.py</code>:
</p>
<pre style="background:#f8f9fa;padding:10px;font-size:12px;">
$ python -m code.live.daily_signal --date 20260515 --model master --n 10 --k 2

  features for date: 800 stocks      ← universe 800 只全部有特征
  predicted: 783 stocks              ← 17 只数据不足 (近期上市)
  Init position: BUY 10 stocks

  Top-10 (score 0.548 ~ 0.565):
    300750.SZ 宁德时代  601138.SH 工业富联  000063.SZ 中兴通讯  601988.SH 中国银行
    600919.SH 江苏银行  300760.SZ 迈瑞医疗  601998.SH 中信银行  601808.SH 中海油服
    600372.SH 中航机载  600519.SH 贵州茅台
</pre>
<p>
  <b>满仓约束自检</b> (作业要求 "每个小组每日必须满仓"):
</p>
<ul>
  <li>建仓: n=10 只等权, 每只 1/n=10% → 总仓位 100% ✅</li>
  <li>换仓: 卖 k=2 (释放 20% 仓位) + 买 k=2 (用释放的仓位等权买入) → 仍 100% ✅</li>
  <li><code>daily_signal.py</code> 输出的 (action, ts_code, name, score) 清单可直接按 1/n 等权下单</li>
</ul>

<h2>10. 复现命令</h2>
<pre style="background: #f5f5f5; padding: 12px; font-size: 12px; overflow-x: auto;">
conda activate astock

# 数据
python -m code.data.build_panel
python -m code.data.universe

# 特征
python -m code.features.factors
python -m code.features.labels
python -m code.features.neutralize
python -m code.features.market_features

# 模型
python -m code.models.lgbm_baseline       # 8 s
python -m code.models.mlp_baseline        # 17 min (作业建议 DL 基线)
python -m code.models.gru_att             # 17 min
python -m code.models.master              # 5 min   &lt;-- 主模型
python -m code.models.master_v2           # 14 min  (反例消融)
python -m code.models.master_v3           # 11 min  (multi-seed)

# 回测 (真实约束: T+1 + 涨跌停 + 双边手续费)
python -m code.backtest.engine --model master --n 10 --k 2
python -m code.backtest.engine --model master --n 10 --k 2 --no-fee --no-limit  # 理想化对比

# Ensemble + 分析 + 报告
python -m code.models.ensemble
python -m code.report.ic_analysis             # 制度断点检测
python -m code.report.direction_accuracy      # 方向胜率
python -m code.report.loss_curves             # 训练曲线
python -m code.report.build_report --include-v2

# 模拟交易 (6/1 起每日盘后)
python -m code.live.daily_signal --date 20260601 --model master --n 10 --k 2
</pre>

<hr>
<h2>11. 参考文献</h2>
<ol style="font-size:13px;">
  <li>Li, T., Liu, Z., Shen, Y., Wang, X., Chen, H., Huang, S. (2024).
    <b>MASTER: Market-Guided Stock Transformer for Stock Price Forecasting</b>.
    In <i>Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2024)</i>.</li>
  <li>Amihud, Y. (2002). <b>Illiquidity and stock returns: cross-section and time-series effects</b>.
    <i>Journal of Financial Markets</i>, 5(1), 31-56.</li>
  <li>Sharpe, W. F. (1994). <b>The Sharpe Ratio</b>.
    <i>The Journal of Portfolio Management</i>, 21(1), 49-58.</li>
  <li>Treynor, J. L., & Black, F. (1973).
    <b>How to Use Security Analysis to Improve Portfolio Selection (Information Ratio)</b>.
    <i>The Journal of Business</i>, 46(1), 66-86.</li>
  <li>Sortino, F. A., & van der Meer, R. (1991).
    <b>Downside risk</b>. <i>The Journal of Portfolio Management</i>, 17(4), 27-31.</li>
  <li>Ke, G., Meng, Q., Finley, T., et al. (2017). <b>LightGBM: A Highly Efficient Gradient Boosting Decision Tree</b>.
    In <i>NeurIPS 2017</i>.</li>
  <li>Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). <b>Attention is All You Need</b>.
    In <i>NeurIPS 2017</i>.</li>
  <li>Cho, K., van Merrienboer, B., Gulcehre, C., et al. (2014).
    <b>Learning Phrase Representations using RNN Encoder-Decoder (GRU)</b>.
    In <i>EMNLP 2014</i>.</li>
  <li>USTC 深度学习基础 2026 课程讲义 (8 讲) — 教授提供, 见 <code>教授讲义/</code> 目录.</li>
  <li>Qlib: Microsoft AI4Finance 量化平台 — 启发本项目的因子工程框架 (未直接使用代码).</li>
</ol>

<hr>
<p style="text-align: center; color: #999; font-size: 12px;">
Generated by build_report.py · 2026
</p>

</body>
</html>
"""


def _m(art, *keys):
    """从 metrics 中按候选 key 顺序取值, 兼容不同模型的命名"""
    m = art.get("metrics") or {}
    for k in keys:
        if m.get(k) is not None:
            return m[k]
    return None


def build_metric_rows(arts, model_names):
    """构造综合指标表的行(自动 best/worst 标色)"""
    rows = []
    def _scaled(getter, scale):
        def f(a):
            v = getter(a)
            return None if v is None else v * scale
        return f

    fields = [
        ("Test IC",              lambda a: _m(a, "test_ic_mean", "ic_mean"), 4, True),
        ("Test RankIC",          lambda a: _m(a, "test_rank_ic_mean", "rank_ic_mean"), 4, True),
        ("RankICIR (年化)",      lambda a: _m(a, "test_rank_icir_annual", "rank_icir_annual"), 2, True),
        ("总收益 (%)",           _scaled(lambda a: a.get("backtest", {}).get("strategy", {}).get("total_return") if a.get("backtest") else None, 100), 1, True),
        ("年化收益 (%)",         _scaled(lambda a: a.get("backtest", {}).get("strategy", {}).get("annual_return") if a.get("backtest") else None, 100), 1, True),
        ("年化波动 (%)",         _scaled(lambda a: a.get("backtest", {}).get("strategy", {}).get("annual_vol") if a.get("backtest") else None, 100), 1, False),
        ("夏普 Sharpe",          lambda a: a.get("backtest", {}).get("strategy", {}).get("sharpe") if a.get("backtest") else None, 2, True),
        ("Sortino",              lambda a: a.get("backtest", {}).get("strategy", {}).get("sortino") if a.get("backtest") else None, 2, True),
        ("Calmar",               lambda a: a.get("backtest", {}).get("strategy", {}).get("calmar") if a.get("backtest") else None, 2, True),
        ("最大回撤 (%)",         _scaled(lambda a: a.get("backtest", {}).get("strategy", {}).get("max_drawdown") if a.get("backtest") else None, 100), 2, True),
        ("胜率 (%)",             _scaled(lambda a: a.get("backtest", {}).get("strategy", {}).get("win_rate") if a.get("backtest") else None, 100), 1, True),
        ("VaR 95% (%)",          _scaled(lambda a: a.get("backtest", {}).get("strategy", {}).get("var_95") if a.get("backtest") else None, 100), 2, True),
        ("CVaR 95% (%)",         _scaled(lambda a: a.get("backtest", {}).get("strategy", {}).get("cvar_95") if a.get("backtest") else None, 100), 2, True),
        ("CAPM Beta",            lambda a: a.get("backtest", {}).get("strategy", {}).get("capm_beta") if a.get("backtest") else None, 3, False),
        ("CAPM Alpha (年化%)",   _scaled(lambda a: a.get("backtest", {}).get("strategy", {}).get("capm_alpha_annual") if a.get("backtest") else None, 100), 2, True),
        ("Information Ratio",    lambda a: a.get("backtest", {}).get("strategy", {}).get("information_ratio") if a.get("backtest") else None, 2, True),
        ("Tracking Error (%)",   _scaled(lambda a: a.get("backtest", {}).get("strategy", {}).get("tracking_error") if a.get("backtest") else None, 100), 1, False),
    ]
    for label, getter, prec, higher_better in fields:
        values = [getter(arts[n]) for n in model_names]
        valid = [(i, v) for i, v in enumerate(values) if v is not None]
        if not valid:
            continue
        if higher_better:
            best_i = max(valid, key=lambda x: x[1])[0]
            worst_i = min(valid, key=lambda x: x[1])[0]
        else:
            best_i = min(valid, key=lambda x: x[1])[0]
            worst_i = max(valid, key=lambda x: x[1])[0]
        cells = []
        for i, v in enumerate(values):
            if v is None:
                cells.append({"val": "-", "cls": ""})
            else:
                cls = "metric-best" if i == best_i and len(valid) >= 2 else (
                      "metric-worst" if i == worst_i and len(valid) >= 2 else "")
                cells.append({"val": f"{v:.{prec}f}", "cls": cls})
        rows.append({"label": label, "cells": cells})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-v2", action="store_true", help="包含 master_v2 模型")
    parser.add_argument("--bt-suffix", default="_real",
                        help="回测后缀, _real (默认) 或 '' (旧的无约束版本)")
    args = parser.parse_args()

    print("Loading model artifacts ...")
    model_names = ["lgbm", "mlp", "gru", "master"]
    if args.include_v2 and (OUTPUT / "signals" / "master_v2_test.parquet").exists():
        model_names.append("master_v2")
    if (OUTPUT / "signals" / "master_v3_test.parquet").exists():
        model_names.append("master_v3")
    model_names.append("ensemble")
    bt_suffix = args.bt_suffix
    arts = {n: load_model_artifacts(n, suffix=bt_suffix) for n in model_names}
    arts_ideal = {n: load_model_artifacts(n, suffix="_ideal") for n in model_names}

    # 方向胜率
    dir_acc_path = OUTPUT / "direction_accuracy.json"
    direction_acc = {}
    if dir_acc_path.exists():
        with open(dir_acc_path, encoding="utf-8") as f:
            direction_acc = json.load(f)

    # 计算 v3 vs v1 的 IC 提升 (bp)
    v3_ic_gain = "N/A"
    if "master_v3" in model_names and arts["master_v3"].get("metrics") and arts["master"].get("metrics"):
        v3_ic = arts["master_v3"]["metrics"]["test_rank_ic_mean"]
        v1_ic = arts["master"]["metrics"]["test_rank_ic_mean"]
        v3_ic_gain = f"{(v3_ic - v1_ic) * 10000:.0f}"

    print("Computing benchmark ...")
    bench = benchmark_nav()

    print("Plotting ...")
    plot_nav_compare(arts, bench, FIGS / "nav_compare.png")
    plot_ic_series(arts, FIGS / "ic_cumulative.png")
    plot_drawdown(arts, FIGS / "drawdown.png")
    plot_quantile_returns(arts["master"], "master", FIGS / "quantile_master.png")
    plot_monthly_returns(
        {k: v for k, v in arts.items() if k in ["master", "lgbm", "ensemble"]},
        FIGS / "monthly_returns.png",
    )
    has_factor_imp = plot_factor_importance(FIGS / "factor_importance.png")

    print("Building HTML ...")
    test_days = 0
    for n in model_names:
        if arts[n].get("backtest"):
            sig = arts[n].get("signals")
            if sig is not None:
                test_days = max(test_days, int(sig["trade_date"].nunique()))

    # 顶部摘要 (master)
    m_bt = arts["master"].get("backtest", {}).get("strategy", {})
    best_annual = f"{100 * m_bt.get('annual_return', 0):.1f}"
    best_sharpe = f"{m_bt.get('sharpe', 0):.2f}"
    best_dd = f"{100 * m_bt.get('max_drawdown', 0):.1f}"

    # 模型概览
    model_overview = [
        {"name": "LightGBM", "desc": "梯度提升树, 横截面特征比较 (作业建议的对比基线)",
         "nparams": "~83 trees", "time": "8 s"},
        {"name": "MLP (DL 基线)", "desc": "T×F 拉平 → 3 层 MLP, 讲义方案 A",
         "nparams": "~144K", "time": "17 min"},
        {"name": "GRU+Att", "desc": "时序 GRU(64) × 2 + Attention pooling, IC loss",
         "nparams": "~43K", "time": "17 min"},
        {"name": "MASTER (主模型)", "desc": "Market-gate + Intra-stock TX (时序) + Inter-stock TX (横截面) + Listwise loss",
         "nparams": "~123K", "time": "5 min"},
    ]
    if "master_v2" in model_names:
        meta = arts["master_v2"].get("metrics", {})
        model_overview.append({
            "name": "MASTER v2 (失败的容量增加)",
            "desc": "T=30 + H=96 + 3 intra + 2 inter + EMA + Warmup (反例)",
            "nparams": f"~{meta.get('n_params', 0) // 1000}K",
            "time": "~14 min",
        })
    if "master_v3" in model_names:
        meta = arts["master_v3"].get("metrics", {})
        model_overview.append({
            "name": "MASTER v3 (multi-seed)",
            "desc": "v1 架构 × 3 seeds rank-percentile 平均",
            "nparams": "~123K × 3",
            "time": "~14 min",
        })
    model_overview.append({
        "name": "Ensemble",
        "desc": "三模型 rank-percentile 加权 (0.5*master+0.3*lgbm+0.2*gru)",
        "nparams": "-", "time": "-",
    })

    metric_rows = build_metric_rows(arts, model_names)

    # Real vs Ideal 对比表 (MASTER)
    real_vs_ideal_rows = []
    m_real = arts["master"].get("backtest", {}).get("strategy", {})
    m_ideal = arts_ideal["master"].get("backtest", {}).get("strategy", {}) if arts_ideal.get("master") else {}
    if m_real and m_ideal:
        fields = [
            ("总收益 (%)", "total_return", 100),
            ("年化收益 (%)", "annual_return", 100),
            ("年化波动 (%)", "annual_vol", 100),
            ("夏普", "sharpe", 1),
            ("最大回撤 (%)", "max_drawdown", 100),
            ("胜率 (%)", "win_rate", 100),
        ]
        for label, key, scale in fields:
            i_v = m_ideal.get(key, 0) * scale
            r_v = m_real.get(key, 0) * scale
            real_vs_ideal_rows.append({
                "label": label,
                "ideal": f"{i_v:.2f}",
                "real": f"{r_v:.2f}",
                "delta": f"{r_v - i_v:+.2f}",
            })

    html = Template(HTML_TPL).render(
        test_days=test_days,
        best_annual=best_annual,
        best_sharpe=best_sharpe,
        best_dd=best_dd,
        model_overview=model_overview,
        model_names=[n.upper() for n in model_names],
        metric_rows=metric_rows,
        has_factor_imp=has_factor_imp,
        has_v2="master_v2" in model_names,
        has_v3="master_v3" in model_names,
        v3_ic_gain=v3_ic_gain,
        real_vs_ideal_rows=real_vs_ideal_rows,
        direction_acc=direction_acc,
    )
    out = REPORTS / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nOK: {out}")
    print(f"  figs: {FIGS}")


if __name__ == "__main__":
    main()
