"""
日频回测引擎 v2 (真实交易约束):
  - 持仓 n 只, 每日换手 k 只
  - t 日 score 决定 t+1 持仓 (隐式 T+1: portfolio 是上一日 score 决定的)
  - 持仓收益用 close-to-close (pct_chg)
  - 新增: 涨跌停 (一字板) 过滤 + 手续费扣除
  - ST 当日动态剔除

手续费 (作业讲义建议):
  买入: 0.025% 佣金
  卖出: 0.025% 佣金 + 0.1% 印花税 = 0.125%

Usage:
  python -m code.backtest.engine --model master --n 10 --k 2          # 真实约束 (有费+限板)
  python -m code.backtest.engine --model master --n 10 --k 2 --no-fee --no-limit  # 理想化对比
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
OUTPUT = ROOT / "output"
(OUTPUT / "reports").mkdir(parents=True, exist_ok=True)

FEE_BUY = 0.00025   # 0.025% 佣金
FEE_SELL = 0.00125  # 0.025% 佣金 + 0.1% 印花税


def load_signals_and_panel(model_name):
    sig_path = OUTPUT / "signals" / f"{model_name}_test.parquet"
    signals = pd.read_parquet(sig_path)
    print(f"Loaded signals: {sig_path}  shape={signals.shape}")

    panel = pd.read_parquet(
        CACHE / "panel.parquet",
        columns=["trade_date", "ts_code", "open", "high", "low", "close",
                 "pct_chg", "is_st"],
    )
    panel = panel[panel["trade_date"] >= 20231201]
    return signals, panel


def backtest(signals, panel, n=10, k=2, start_date=20240101,
             apply_fee=True, apply_limit=True):
    """
    回测引擎:
      - 每日: 1) 用昨日 portfolio 算今日 pct_chg 收益
              2) 用今日 score 决定明日卖出 (跌停过滤) / 买入 (涨停过滤)
              3) 扣手续费
    """
    panel_pivot = panel.pivot_table(
        index="trade_date", columns="ts_code", values="pct_chg"
    ).sort_index() / 100.0

    st_pivot = panel.pivot_table(
        index="trade_date", columns="ts_code", values="is_st"
    ).sort_index().fillna(False).astype(bool)

    open_p = panel.pivot_table(index="trade_date", columns="ts_code", values="open").sort_index()
    high_p = panel.pivot_table(index="trade_date", columns="ts_code", values="high").sort_index()
    low_p  = panel.pivot_table(index="trade_date", columns="ts_code", values="low").sort_index()
    # 一字板 mask: open == high == low
    yzi_mask = (open_p == high_p) & (open_p == low_p)

    signals = signals.sort_values(["trade_date", "score"], ascending=[True, False])
    dates = sorted(signals["trade_date"].unique())
    dates = [d for d in dates if d >= start_date]

    portfolio = set()
    nav = 1.0
    nav_list = []
    daily_returns = []
    trades = []
    fee_total = 0.0

    sig_groups = signals.groupby("trade_date")

    for i, t_date in enumerate(dates):
        sig_t = sig_groups.get_group(t_date)

        # 当日 ST 剔除 (买入/卖出候选都不能是 ST)
        if t_date in st_pivot.index:
            st_today = st_pivot.loc[t_date]
            st_set = set(st_today[st_today].index.tolist())
            sig_t = sig_t[~sig_t["ts_code"].isin(st_set)]
        else:
            st_set = set()

        # 当日一字板
        if apply_limit and t_date in yzi_mask.index:
            yzi_today = yzi_mask.loc[t_date]
            yzi_set = set(yzi_today[yzi_today].index.tolist())
        else:
            yzi_set = set()

        # 1. 首日建仓
        if not portfolio:
            buy_candidates = sig_t[~sig_t["ts_code"].isin(yzi_set)]
            buy_list = buy_candidates.head(n)["ts_code"].tolist()
            portfolio.update(buy_list)
            nav_list.append(nav)
            daily_returns.append(0.0)
            if apply_fee:
                # 建仓买入费: 每只 1/n 仓位 × FEE_BUY
                init_fee = FEE_BUY  # 总仓位 1.0 × FEE_BUY
                nav *= (1 - init_fee)
                fee_total += init_fee
            trades.append({"date": t_date, "action": "init",
                           "n_buy": len(buy_list), "nav": nav})
            continue

        # 2. 计算上日 portfolio 在今日的 pct_chg 收益 (T+1: portfolio 来自上日 score)
        if t_date in panel_pivot.index:
            day_ret_series = panel_pivot.loc[t_date]
            held_rets = day_ret_series.reindex(list(portfolio)).fillna(0.0).values
            day_ret = float(held_rets.mean())  # 等权
        else:
            day_ret = 0.0

        nav *= (1 + day_ret)
        nav_list.append(nav)
        daily_returns.append(day_ret)

        # 3. 决定换仓: 用 t 日 score
        in_pos_sig = sig_t[sig_t["ts_code"].isin(portfolio)].sort_values("score")
        not_in_sig = sig_t[~sig_t["ts_code"].isin(portfolio)].sort_values(
            "score", ascending=False
        )

        # 卖出: 当前持仓得分最低的 k 只 (跳过一字跌停, 但作为简化, 一字板既不能买也不能卖)
        sells_planned = in_pos_sig["ts_code"].tolist()
        sells = []
        for c in sells_planned:
            if apply_limit and c in yzi_set:
                continue
            sells.append(c)
            if len(sells) >= k:
                break

        # 买入: 非持仓得分最高的, 跳过涨停和 ST
        buys_planned = not_in_sig["ts_code"].tolist()
        buys = []
        n_to_buy = len(sells)  # 卖几只买几只, 保持仓位
        for c in buys_planned:
            if apply_limit and c in yzi_set:
                continue
            buys.append(c)
            if len(buys) >= n_to_buy:
                break

        # 4. 执行 + 扣费
        for c in sells:
            portfolio.discard(c)
        portfolio.update(buys)

        if apply_fee and (sells or buys):
            # 换手率 = (sell + buy) / 2n; 单边费率 = FEE_SELL on sell + FEE_BUY on buy
            sell_w = len(sells) / n
            buy_w = len(buys) / n
            fee_today = sell_w * FEE_SELL + buy_w * FEE_BUY
            nav *= (1 - fee_today)
            fee_total += fee_today

        trades.append({
            "date": t_date, "action": "trade",
            "n_buy": len(buys), "n_sell": len(sells),
            "ret": day_ret, "nav": nav,
            "fee_cum": fee_total,
        })

    nav_series = pd.Series(nav_list, index=dates, name="nav")
    daily_ret_series = pd.Series(daily_returns, index=dates)
    print(f"  Total fee: {fee_total*100:.3f}%  trades: {len(trades)}")
    return nav_series, daily_ret_series, pd.DataFrame(trades)


def benchmark_nav(start_date=20240101):
    bench = pd.read_csv(ROOT / "market" / "000300.SH.csv")
    bench = bench.sort_values("trade_date")
    bench = bench[bench["trade_date"] >= start_date]
    bench["nav"] = (1 + bench["pct_chg"] / 100).cumprod()
    bench["nav"] = bench["nav"] / bench["nav"].iloc[0]
    return bench.set_index("trade_date")["nav"]


def compute_metrics(daily_ret, nav, bench_ret=None):
    """完整金融指标:
      基础: total/annual/vol/sharpe/MDD/win_rate
      下行风险: Sortino, Calmar, VaR95, CVaR95
      相对基准 (若给): CAPM Alpha/Beta, Tracking Error, Information Ratio
    """
    annual_ret = (1 + daily_ret.mean()) ** 252 - 1
    annual_vol = daily_ret.std() * np.sqrt(252)
    sharpe = annual_ret / (annual_vol + 1e-8)
    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (daily_ret > 0).sum() / len(daily_ret)

    # Sortino: 只考虑下行波动
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = annual_ret / (downside_vol + 1e-8)

    # Calmar: annual_return / |max_dd|
    calmar = annual_ret / (abs(max_dd) + 1e-8)

    # VaR / CVaR (95%)
    var95 = float(np.percentile(daily_ret, 5))
    cvar95 = float(daily_ret[daily_ret <= var95].mean()) if (daily_ret <= var95).any() else var95

    out = {
        "total_return": float(nav.iloc[-1] - 1),
        "annual_return": float(annual_ret),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "var_95": var95,
        "cvar_95": cvar95,
    }

    if bench_ret is not None and len(bench_ret) == len(daily_ret):
        # CAPM 回归: r_strat = alpha + beta * r_bench + epsilon
        x = bench_ret.values
        y = daily_ret.values
        x_c = x - x.mean()
        y_c = y - y.mean()
        beta = float((x_c * y_c).sum() / ((x_c ** 2).sum() + 1e-12))
        alpha_daily = float(y.mean() - beta * x.mean())
        alpha_annual = (1 + alpha_daily) ** 252 - 1

        # Information Ratio
        excess = daily_ret - bench_ret
        tracking_error = excess.std() * np.sqrt(252)
        info_ratio = (excess.mean() * 252) / (tracking_error + 1e-8)

        out["capm_beta"] = beta
        out["capm_alpha_annual"] = float(alpha_annual)
        out["tracking_error"] = float(tracking_error)
        out["information_ratio"] = float(info_ratio)
        out["excess_annual"] = float(annual_ret - ((1 + bench_ret.mean()) ** 252 - 1))

    return out


def plot_nav(strat_nav, bench_nav, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(strat_nav)), strat_nav.values, label="Strategy", lw=2)
    bench_aligned = bench_nav.reindex(strat_nav.index)
    bench_aligned = bench_aligned / bench_aligned.iloc[0]
    ax.plot(range(len(bench_aligned)), bench_aligned.values, label="HS300",
            lw=1.5, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Days")
    ax.set_ylabel("NAV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  saved plot: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm",
                        choices=["lgbm", "mlp", "gru", "master", "master_v2", "master_v3", "ensemble"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--start", type=int, default=20240101)
    parser.add_argument("--no-fee", action="store_true")
    parser.add_argument("--no-limit", action="store_true")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    apply_fee = not args.no_fee
    apply_limit = not args.no_limit
    if args.suffix:
        suffix = args.suffix
    elif not apply_fee and not apply_limit:
        suffix = "_ideal"
    elif apply_fee and apply_limit:
        suffix = "_real"
    else:
        suffix = ""

    signals, panel = load_signals_and_panel(args.model)
    print(f"\nBacktest {args.model.upper()}: n={args.n}, k={args.k}, "
          f"fee={apply_fee}, limit={apply_limit}")

    nav, daily_ret, trades = backtest(
        signals, panel, n=args.n, k=args.k, start_date=args.start,
        apply_fee=apply_fee, apply_limit=apply_limit,
    )

    bench_nav = benchmark_nav(start_date=args.start)
    bench_aligned = bench_nav.reindex(nav.index).ffill().fillna(1.0)
    bench_ret = bench_aligned.pct_change().fillna(0)
    bench_metrics = compute_metrics(bench_ret, bench_aligned / bench_aligned.iloc[0])
    metrics = compute_metrics(daily_ret, nav, bench_ret=bench_ret.reindex(daily_ret.index).fillna(0))

    print("\n=== Strategy ===")
    for kk, vv in metrics.items():
        print(f"  {kk:18s} {vv:.4f}")
    print("\n=== HS300 Benchmark ===")
    for kk, vv in bench_metrics.items():
        print(f"  {kk:18s} {vv:.4f}")
    print(f"\n  excess_annual      {metrics['annual_return'] - bench_metrics['annual_return']:.4f}")

    nav.to_csv(OUTPUT / "reports" / f"backtest_{args.model}{suffix}_nav.csv", header=["nav"])
    with open(OUTPUT / f"backtest_{args.model}{suffix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "strategy": metrics,
            "benchmark": bench_metrics,
            "excess_annual": metrics["annual_return"] - bench_metrics["annual_return"],
            "n_holdings": args.n,
            "k_swap": args.k,
            "start_date": args.start,
            "apply_fee": apply_fee,
            "apply_limit": apply_limit,
            "fee_buy_pct": FEE_BUY if apply_fee else 0,
            "fee_sell_pct": FEE_SELL if apply_fee else 0,
        }, f, indent=2)
    plot_nav(nav, bench_nav,
             f"{args.model.upper()} (n={args.n},k={args.k},fee={apply_fee},limit={apply_limit})",
             OUTPUT / "reports" / f"backtest_{args.model}{suffix}.png")


if __name__ == "__main__":
    main()
