"""
中证800近似选股池: 每月初取自由流通市值前800
过滤: 非ST + 上市>=180天 + 有交易量 + 流通市值>0
Output: cache/universe.parquet (yyyymm, trade_date, ts_code, circ_mv)
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"


def build(top_n=800):
    print("Loading panel ...")
    panel = pd.read_parquet(
        CACHE / "panel.parquet",
        columns=["trade_date", "ts_code", "is_st", "list_days", "circ_mv", "vol"],
    )
    panel["yyyymm"] = panel["trade_date"] // 100  # 20160115 -> 201601

    # 每月首交易日
    month_first = panel.groupby("yyyymm")["trade_date"].min()
    print(f"Months: {len(month_first)} ({month_first.iloc[0]} ~ {month_first.iloc[-1]})")

    rows = []
    for yyyymm, date in tqdm(month_first.items(), total=len(month_first), desc="select"):
        snap = panel[panel["trade_date"] == date]
        mask = (
            (~snap["is_st"])
            & (snap["list_days"].fillna(-1) >= 180)
            & (snap["circ_mv"].fillna(0) > 0)
            & (snap["vol"].fillna(0) > 0)
        )
        cand = snap[mask].nlargest(top_n, "circ_mv")
        rows.append(pd.DataFrame({
            "yyyymm": yyyymm,
            "trade_date": date,
            "ts_code": cand["ts_code"].values,
            "circ_mv": cand["circ_mv"].values,
        }))

    universe = pd.concat(rows, ignore_index=True)
    out = CACHE / "universe.parquet"
    universe.to_parquet(out, index=False)

    print(f"\n{'='*60}")
    print(f"OK saved: {out}")
    print(f"  rows           : {len(universe):,}")
    print(f"  months         : {universe['yyyymm'].nunique()}")
    print(f"  unique stocks  : {universe['ts_code'].nunique()}")
    print(f"  avg per month  : {len(universe) / universe['yyyymm'].nunique():.0f}")
    # 月际变化(换仓率)
    monthly = universe.groupby("yyyymm")["ts_code"].apply(set)
    if len(monthly) > 1:
        turnovers = []
        prev = None
        for s in monthly:
            if prev is not None:
                turnovers.append(len(s - prev) / len(s))
            prev = s
        print(f"  avg turnover   : {sum(turnovers)/len(turnovers)*100:.1f}% per month")


if __name__ == "__main__":
    build()
