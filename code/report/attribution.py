"""
行业 / 市值归因分析:
  - 持仓的行业 / 市值分布
  - 不同行业 / 市值组的收益贡献
Output:
  output/reports/figs/attribution_industry.png
  output/reports/figs/attribution_marketcap.png
  output/attribution.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
FIGS = OUTPUT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_master_top10():
    """逐日取 master signals 的 top-10 作为模拟持仓"""
    sig = pd.read_parquet(OUTPUT / "signals" / "master_test.parquet")
    sig = sig.sort_values(["trade_date", "score"], ascending=[True, False])
    top = sig.groupby("trade_date").head(10).reset_index(drop=True)
    return top


def industry_attribution():
    top = load_master_top10()
    basic = pd.read_csv(ROOT / "basic.csv")[["ts_code", "industry"]]
    panel = pd.read_parquet(CACHE / "panel.parquet",
                              columns=["trade_date", "ts_code", "pct_chg", "circ_mv"])
    panel = panel[panel["trade_date"] >= 20231201]

    # 计算 forward 1-day return: 用 (date, ts_code) 的 pct_chg 但 shift 到上一天
    # 即: top@t 应该获取 t+1 的 pct_chg
    panel = panel.sort_values(["ts_code", "trade_date"])
    panel["fwd_pct_chg"] = panel.groupby("ts_code")["pct_chg"].shift(-1)
    panel = panel[["trade_date", "ts_code", "fwd_pct_chg", "circ_mv"]]

    df = top.merge(basic, on="ts_code", how="left")
    df = df.merge(panel, on=["trade_date", "ts_code"], how="left")
    df["fwd_pct_chg"] = df["fwd_pct_chg"] / 100.0  # %
    df = df.dropna(subset=["fwd_pct_chg"])

    # 行业占比 (按持仓次数)
    ind_count = df["industry"].value_counts()
    total = len(df)
    ind_share = (ind_count / total * 100).round(2)
    # 行业贡献 (这些位置的平均 forward 日收益)
    ind_ret = df.groupby("industry")["fwd_pct_chg"].mean() * 100  # %

    # 市值分组 (5 档)
    df["mv_bucket"] = pd.qcut(df["circ_mv"], 5,
                                labels=["XS (微盘)", "S", "M", "L", "XL (大盘)"],
                                duplicates="drop")
    mv_share = df["mv_bucket"].value_counts(normalize=True) * 100
    mv_ret = df.groupby("mv_bucket", observed=False)["fwd_pct_chg"].mean() * 100

    return {
        "ind_share": ind_share.head(20).to_dict(),
        "ind_ret": ind_ret.to_dict(),
        "mv_share": mv_share.to_dict(),
        "mv_ret": mv_ret.to_dict(),
        "total_holdings": int(total),
        "n_industries": int(df["industry"].nunique()),
        "comment": "用持仓日 t+1 (forward 1d) 收益归因, 正确反映策略表现",
    }


def plot_industry(att, save):
    ind_share = pd.Series(att["ind_share"]).sort_values(ascending=True).tail(15)
    ind_ret = pd.Series(att["ind_ret"]).reindex(ind_share.index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax = axes[0]
    colors = ["#4488cc"] * len(ind_share)
    ax.barh(ind_share.index, ind_share.values, color=colors)
    ax.set_xlabel("持仓占比 (%)")
    ax.set_title("MASTER 持仓行业分布 (Top 15)")
    for i, v in enumerate(ind_share.values):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1]
    colors = ["#33aa66" if v > 0 else "#cc4433" for v in ind_ret.values]
    ax.barh(ind_ret.index, ind_ret.values, color=colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("平均日收益 (%)")
    ax.set_title("各行业持仓日收益均值")
    for i, v in enumerate(ind_ret.values):
        ax.text(v + 0.02 * (1 if v > 0 else -1), i,
                f"{v:.2f}%", va="center", fontsize=9,
                ha="left" if v > 0 else "right")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(save, dpi=120); plt.close(fig)


def plot_marketcap(att, save):
    order = ["XS (微盘)", "S", "M", "L", "XL (大盘)"]
    mv_share = pd.Series(att["mv_share"]).reindex(order).fillna(0)
    mv_ret = pd.Series(att["mv_ret"]).reindex(order).fillna(0)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    w = 0.35
    b1 = ax.bar(x - w/2, mv_share.values, w, label="持仓占比 (%)", color="#4488cc")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w/2, mv_ret.values * 100, w, label="日收益均值 (bp)",
                  color="#cc6644")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("持仓占比 (%)", color="#4488cc")
    ax2.set_ylabel("日收益均值 (bp)", color="#cc6644")
    ax.set_title("MASTER 持仓的市值分组归因 (5 分位)")
    for i, v in enumerate(mv_share.values):
        ax.text(i - w/2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    for i, v in enumerate(mv_ret.values * 100):
        ax2.text(i + w/2, v + 1, f"{v:.0f}bp", ha="center", fontsize=9, color="#cc6644")
    ax.grid(True, alpha=0.3, axis="y")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2)
    fig.tight_layout()
    fig.savefig(save, dpi=120); plt.close(fig)


def turnover_analysis():
    """从 backtest 引擎重跑出 trades 序列, 计算换手率"""
    from code.backtest.engine import load_signals_and_panel, backtest
    sigs, panel = load_signals_and_panel("master")
    nav, daily_ret, trades = backtest(sigs, panel, n=10, k=2, start_date=20240101,
                                       apply_fee=True, apply_limit=True)
    # n_buy/n_sell 是每日交易笔数
    if "n_buy" in trades.columns:
        avg_n_buy = float(trades["n_buy"].mean())
        avg_n_sell = float(trades["n_sell"].mean()) if "n_sell" in trades.columns else 0
    else:
        avg_n_buy = avg_n_sell = 0
    # 双边换手率 (每日 sell+buy/2n)
    daily_turnover = (avg_n_buy + avg_n_sell) / (2 * 10) * 100  # %
    annual_turnover = daily_turnover * 252  # %
    return {
        "n_trade_days": int(len(trades)),
        "avg_daily_buys": avg_n_buy,
        "avg_daily_sells": avg_n_sell,
        "daily_turnover_pct": daily_turnover,
        "annual_turnover_pct": annual_turnover,
        "comment": "k=2 设计 = 每日换 20% 仓位 (双边); 年化换手 ~50 倍",
    }


def main():
    print("Computing industry / mv attribution ...")
    att = industry_attribution()
    print(f"  Total holdings: {att['total_holdings']:,}, "
          f"unique industries: {att['n_industries']}")

    print("\nTop 行业占比:")
    for k, v in list(att["ind_share"].items())[:10]:
        print(f"  {k}: {v}%")

    print("\n市值分组占比:")
    for k, v in att["mv_share"].items():
        print(f"  {k}: {v:.2f}%, 平均日收益 {att['mv_ret'][k]:.4f}")

    plot_industry(att, FIGS / "attribution_industry.png")
    plot_marketcap(att, FIGS / "attribution_marketcap.png")
    print(f"\nSaved: {FIGS / 'attribution_industry.png'}, {FIGS / 'attribution_marketcap.png'}")

    print("\nTurnover analysis ...")
    turn = turnover_analysis()
    print(f"  Avg daily buys/sells: {turn['avg_daily_buys']:.2f} / {turn['avg_daily_sells']:.2f}")
    print(f"  Daily turnover: {turn['daily_turnover_pct']:.2f}%, "
          f"Annual: {turn['annual_turnover_pct']:.1f}%")

    with open(OUTPUT / "attribution.json", "w", encoding="utf-8") as f:
        json.dump({"attribution": att, "turnover": turn},
                  f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT / 'attribution.json'}")


if __name__ == "__main__":
    main()
