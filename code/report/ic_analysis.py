"""
分段 IC 分析: 检测制度断点候选时点 (教授讲义第八讲核心方法)
  - 训练期 (2016-2023) 按年度切 IC, 看是否稳定
  - 测试期 (2024-2025) 按季度切 IC
  - 用 lgbm + master 两个模型对比 (lgbm 训练快, 能跨年度产生信号)
Output:
  output/reports/figs/ic_by_period.png
  output/reports/figs/ic_by_quarter.png
  output/ic_segment_metrics.json
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from code.config import ROOT, CACHE, OUTPUT

REPORTS = OUTPUT / "reports"
FIGS = REPORTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def daily_ic(df_sig):
    """返回 DataFrame (trade_date, ic, rank_ic) — 用于时序图"""
    from code.metrics import daily_ic as _daily_ic
    out = []
    for d, day in df_sig.groupby("trade_date"):
        if len(day) < 30 or day["score"].std() == 0:
            continue
        out.append((d, day["score"].corr(day["label"]),
                    day["score"].rank().corr(day["label"].rank())))
    return pd.DataFrame(out, columns=["trade_date", "ic", "rank_ic"])


def _to_jsonable(o):
    """递归转换 numpy 类型为 Python 原生类型 (用于 json.dump)"""
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def compute_train_ic_via_lgbm():
    """用已训练 lgbm 模型在全部历史数据上推理, 得到 train+val+test 的 IC 序列"""
    import pickle
    print("Loading lgbm model + full features ...")
    with open(OUTPUT / "checkpoints" / "lgbm.pkl", "rb") as f:
        lgbm = pickle.load(f)
    feats = pd.read_parquet(CACHE / "features.parquet",
                            columns=FACTOR_COLS + ["trade_date", "ts_code"])
    labels = pd.read_parquet(CACHE / "labels.parquet",
                             columns=["trade_date", "ts_code", "label"])

    df = feats.merge(labels[["trade_date", "ts_code", "label"]],
                     on=["trade_date", "ts_code"], how="inner")
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    from code.config import FACTOR_COLS
    X = df[FACTOR_COLS].fillna(0).values.astype(np.float32)
    df["score"] = lgbm.predict(X)
    print(f"  predicted {len(df):,} rows over {df['trade_date'].nunique()} days")
    return daily_ic(df[["trade_date", "score", "label"]])


def by_year(ic_df):
    """聚合到年度 IC"""
    ic_df = ic_df.copy()
    ic_df["year"] = ic_df["trade_date"] // 10000
    agg = ic_df.groupby("year").agg(
        ic_mean=("ic", "mean"),
        ic_std=("ic", "std"),
        rank_ic_mean=("rank_ic", "mean"),
        rank_ic_std=("rank_ic", "std"),
        n_days=("ic", "count"),
    )
    agg["icir_annual"] = agg["ic_mean"] / (agg["ic_std"] + 1e-8) * np.sqrt(252)
    agg["rank_icir_annual"] = agg["rank_ic_mean"] / (agg["rank_ic_std"] + 1e-8) * np.sqrt(252)
    return agg


def by_quarter(ic_df):
    ic_df = ic_df.copy()
    ic_df["q"] = ic_df["trade_date"].apply(
        lambda d: f"{d // 10000}Q{(((d % 10000) // 100) - 1) // 3 + 1}"
    )
    agg = ic_df.groupby("q").agg(
        ic_mean=("ic", "mean"),
        ic_std=("ic", "std"),
        rank_ic_mean=("rank_ic", "mean"),
        rank_ic_std=("rank_ic", "std"),
        n_days=("ic", "count"),
    )
    agg["icir_annual"] = agg["ic_mean"] / (agg["ic_std"] + 1e-8) * np.sqrt(252)
    return agg


def plot_yearly(yearly_lgbm, yearly_master, save):
    """两模型年度 IC 条形图"""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    years = sorted(set(yearly_lgbm.index.tolist() + yearly_master.index.tolist()))
    x = np.arange(len(years))
    w = 0.35

    # IC mean
    ax = axes[0]
    lgbm_v = [yearly_lgbm.loc[y, "ic_mean"] if y in yearly_lgbm.index else np.nan for y in years]
    master_v = [yearly_master.loc[y, "ic_mean"] if y in yearly_master.index else np.nan for y in years]
    ax.bar(x - w/2, lgbm_v, w, label="LGBM", color="#4488cc")
    ax.bar(x + w/2, master_v, w, label="MASTER", color="#cc6644")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel("Mean Daily IC"); ax.set_title("Yearly IC")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    # 标注训练/测试期
    for i, y in enumerate(years):
        if y <= 2022:
            tag, color = "Train", "#888"
        elif y == 2023:
            tag, color = "Valid", "#999"
        else:
            tag, color = "OOS", "#c33"
        ax.text(i, ax.get_ylim()[1] * 0.95, tag, ha="center", color=color, fontsize=9)

    # RankIC mean
    ax = axes[1]
    lgbm_v = [yearly_lgbm.loc[y, "rank_ic_mean"] if y in yearly_lgbm.index else np.nan for y in years]
    master_v = [yearly_master.loc[y, "rank_ic_mean"] if y in yearly_master.index else np.nan for y in years]
    ax.bar(x - w/2, lgbm_v, w, label="LGBM", color="#4488cc")
    ax.bar(x + w/2, master_v, w, label="MASTER", color="#cc6644")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(years)
    ax.set_ylabel("Mean Daily RankIC"); ax.set_xlabel("Year")
    ax.set_title("Yearly RankIC")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save, dpi=120); plt.close(fig)


def plot_quarterly(qtr_master, save):
    """MASTER 测试期季度 IC 条形图 + 累计 NAV 副图"""
    fig, ax = plt.subplots(figsize=(13, 5))
    qtrs = qtr_master.index.tolist()
    x = np.arange(len(qtrs))
    colors = ["#33aa66" if v > 0 else "#cc4433" for v in qtr_master["rank_ic_mean"]]
    ax.bar(x, qtr_master["rank_ic_mean"], color=colors)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(qtr_master["rank_ic_mean"].mean(), color="#888",
               ls="--", label=f"Mean={qtr_master['rank_ic_mean'].mean():.4f}")
    ax.set_xticks(x); ax.set_xticklabels(qtrs, rotation=45, ha="right")
    ax.set_ylabel("Mean RankIC")
    ax.set_title("MASTER: Quarterly RankIC (2024-2025 OOS)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    # 标 N 日数
    for i, (q, row) in enumerate(qtr_master.iterrows()):
        ax.text(i, row["rank_ic_mean"] + 0.002 * (1 if row["rank_ic_mean"] > 0 else -1),
                f"n={row['n_days']:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save, dpi=120); plt.close(fig)


def main():
    print("\n=== Computing LGBM IC over full history (train+val+test) ===")
    lgbm_ic = compute_train_ic_via_lgbm()
    lgbm_yearly = by_year(lgbm_ic)

    print("\n=== Loading MASTER signals (test only - 2024-2025) ===")
    master_sig = pd.read_parquet(OUTPUT / "signals" / "master_test.parquet")
    master_ic = daily_ic(master_sig[["trade_date", "score", "label"]])
    master_yearly = by_year(master_ic)
    master_qtr = by_quarter(master_ic)

    print("\n=== LGBM Yearly IC ===")
    print(lgbm_yearly.to_string())
    print("\n=== MASTER Yearly IC ===")
    print(master_yearly.to_string())
    print("\n=== MASTER Quarterly IC (OOS) ===")
    print(master_qtr.to_string())

    # 检测制度断点候选: 同比 IC 变化 > 30% 的年份
    lgbm_yearly_sorted = lgbm_yearly.sort_index()
    lgbm_yearly_sorted["ic_yoy_delta"] = lgbm_yearly_sorted["ic_mean"].diff()
    breakpoints = lgbm_yearly_sorted[
        lgbm_yearly_sorted["ic_yoy_delta"].abs() > 0.02
    ].index.tolist()
    print(f"\n候选制度断点年份 (IC YoY 变化 > 0.02): {breakpoints}")

    # 画图
    plot_yearly(lgbm_yearly, master_yearly, FIGS / "ic_by_year.png")
    plot_quarterly(master_qtr, FIGS / "ic_by_quarter.png")
    print(f"\nSaved figs: {FIGS / 'ic_by_year.png'}, {FIGS / 'ic_by_quarter.png'}")

    # 保存指标
    metrics = {
        "lgbm_yearly": lgbm_yearly.to_dict("index"),
        "master_yearly": master_yearly.to_dict("index"),
        "master_quarterly": master_qtr.to_dict("index"),
        "breakpoint_candidate_years": breakpoints,
    }
    with open(OUTPUT / "ic_segment_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics), f, indent=2)


if __name__ == "__main__":
    main()
