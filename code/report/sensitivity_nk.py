"""
(n, k) 持仓换手敏感度网格分析:
  - 主模型 MASTER, 跑 (n=5/10/15/20) × (k=1/2/3/5) = 16 组合
  - 真实约束 (有费 + 涨跌停)
  - 输出 sensitivity 热力图 (sharpe / annual_return / max_dd)
Output:
  output/reports/figs/sensitivity_nk.png
  output/sensitivity_nk.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from code.backtest.engine import (
    load_signals_and_panel, backtest, benchmark_nav, compute_metrics
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
REPORTS = OUTPUT / "reports"
FIGS = REPORTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def heatmap(ax, data, n_list, k_list, title, fmt=".2f", cmap="RdYlGn"):
    """data: shape (len(n_list), len(k_list))"""
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(k_list))); ax.set_xticklabels(k_list)
    ax.set_yticks(range(len(n_list))); ax.set_yticklabels(n_list)
    ax.set_xlabel("k (每日换手)"); ax.set_ylabel("n (持仓数)")
    ax.set_title(title)
    for i in range(len(n_list)):
        for j in range(len(k_list)):
            v = data[i, j]
            ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04)


def main():
    n_list = [5, 10, 15, 20]
    k_list = [1, 2, 3, 5]
    model = "master"

    print(f"Loading {model} signals + panel ...")
    signals, panel = load_signals_and_panel(model)
    bench_nav = benchmark_nav(start_date=20240101)

    sharpe_mat = np.zeros((len(n_list), len(k_list)))
    annual_mat = np.zeros_like(sharpe_mat)
    mdd_mat = np.zeros_like(sharpe_mat)
    ir_mat = np.zeros_like(sharpe_mat)
    calmar_mat = np.zeros_like(sharpe_mat)

    results = []
    for i, n in enumerate(n_list):
        for j, k in enumerate(k_list):
            if k > n:  # 换手数不能超过持仓数
                sharpe_mat[i, j] = np.nan
                annual_mat[i, j] = np.nan
                mdd_mat[i, j] = np.nan
                ir_mat[i, j] = np.nan
                calmar_mat[i, j] = np.nan
                continue
            print(f"  Running n={n}, k={k} ...")
            nav, daily_ret, _ = backtest(signals, panel, n=n, k=k,
                                          start_date=20240101,
                                          apply_fee=True, apply_limit=True)
            bench_aligned = bench_nav.reindex(nav.index).ffill().fillna(1.0)
            bench_ret = bench_aligned.pct_change().fillna(0)
            m = compute_metrics(daily_ret, nav, bench_ret=bench_ret.reindex(daily_ret.index).fillna(0))
            sharpe_mat[i, j] = m["sharpe"]
            annual_mat[i, j] = m["annual_return"] * 100  # %
            mdd_mat[i, j] = m["max_drawdown"] * 100      # %
            ir_mat[i, j] = m.get("information_ratio", np.nan)
            calmar_mat[i, j] = m["calmar"]
            results.append({"n": n, "k": k, **m})

    # 画 2×2 热力图
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    heatmap(axes[0, 0], sharpe_mat, n_list, k_list, "Sharpe (越大越好)", fmt=".2f", cmap="RdYlGn")
    heatmap(axes[0, 1], annual_mat, n_list, k_list, "年化收益 % (越大越好)", fmt=".1f", cmap="RdYlGn")
    heatmap(axes[1, 0], mdd_mat, n_list, k_list, "最大回撤 % (越接近 0 越好)", fmt=".1f", cmap="RdYlGn_r")
    heatmap(axes[1, 1], calmar_mat, n_list, k_list, "Calmar = annual/|MDD|", fmt=".2f", cmap="RdYlGn")
    fig.suptitle(f"MASTER 真实约束: (n, k) 敏感度热力图 (空格=k>n 无效)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS / "sensitivity_nk.png", dpi=120); plt.close(fig)
    print(f"\nSaved: {FIGS / 'sensitivity_nk.png'}")

    # 排名表
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)
    print("\nTop 5 (按 Sharpe):")
    print(df[["n", "k", "annual_return", "sharpe", "calmar", "max_drawdown", "information_ratio"]].head().to_string(index=False))

    with open(OUTPUT / "sensitivity_nk.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    df.to_csv(OUTPUT / "sensitivity_nk.csv", index=False)


if __name__ == "__main__":
    main()
