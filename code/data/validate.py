"""
数据正确性验证: 抽样对比 panel.parquet 与原始 csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"


def check_value_match():
    """[1] 抽查 平安银行 2020-01-15 数据, 对比原始CSV"""
    print("\n[1] Value match: 000001.SZ 2020-01-15")
    panel = pd.read_parquet(CACHE / "panel.parquet")
    row = panel[(panel["ts_code"] == "000001.SZ") & (panel["trade_date"] == 20200115)]
    assert len(row) == 1, f"expected 1 row, got {len(row)}"
    r = row.iloc[0]
    # 对比 daily/20200115.csv
    raw = pd.read_csv(ROOT / "daily" / "20200115.csv")
    raw_row = raw[raw["ts_code"] == "000001.SZ"].iloc[0]
    for col in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
        diff = abs(float(r[col]) - float(raw_row[col]))
        status = "OK" if diff < 0.01 else "FAIL"
        print(f"  {col:12s} panel={r[col]:.4f}  raw={raw_row[col]:.4f}  diff={diff:.6f}  [{status}]")


def check_pct_chg():
    """[2] pct_chg = (close - pre_close) / pre_close * 100"""
    print("\n[2] pct_chg formula validation")
    panel = pd.read_parquet(CACHE / "panel.parquet",
                            columns=["close", "pre_close", "pct_chg"])
    panel = panel.dropna(subset=["close", "pre_close", "pct_chg"])
    panel = panel[panel["pre_close"] > 0].sample(10000, random_state=42)
    computed = (panel["close"] - panel["pre_close"]) / panel["pre_close"] * 100
    diff = (computed - panel["pct_chg"]).abs()
    print(f"  sample=10000, max_diff={diff.max():.4f}, mean_diff={diff.mean():.6f}")
    print(f"  [{'OK' if diff.max() < 0.05 else 'FAIL'}]")


def check_st_consistency():
    """[3] ST 标记数量与原始 stock_st/*.csv 一致"""
    print("\n[3] ST flag count")
    panel = pd.read_parquet(CACHE / "panel.parquet", columns=["trade_date", "ts_code", "is_st"])
    for date in [20200115, 20220615, 20240301]:
        date_str = str(date)
        st_file = ROOT / "stock_st" / f"{date_str}.csv"
        if not st_file.exists():
            print(f"  {date}: ST file not found, skip")
            continue
        raw_count = len(pd.read_csv(st_file))
        panel_count = panel[(panel["trade_date"] == date) & panel["is_st"]].shape[0]
        status = "OK" if abs(raw_count - panel_count) <= 2 else "FAIL"
        print(f"  {date}: raw={raw_count}  panel={panel_count}  [{status}]")


def check_universe_content():
    """[4] universe 中包含明显的大盘股(贵州茅台/工商银行/平安银行)"""
    print("\n[4] Universe content (should contain large caps)")
    uni = pd.read_parquet(CACHE / "universe.parquet")
    must_have = {"600519.SH": "贵州茅台", "601398.SH": "工商银行", "000001.SZ": "平安银行"}
    latest_yyyymm = uni["yyyymm"].max()
    latest = set(uni[uni["yyyymm"] == latest_yyyymm]["ts_code"])
    for code, name in must_have.items():
        status = "OK" if code in latest else "FAIL"
        print(f"  {code} ({name}) in latest({latest_yyyymm}): [{status}]")
    print(f"  total in latest month: {len(latest)}")


def check_date_continuity():
    """[5] 日期轴连续性: 缺失交易日检查"""
    print("\n[5] Date continuity")
    panel = pd.read_parquet(CACHE / "panel.parquet", columns=["trade_date"])
    cal = pd.read_csv(ROOT / "trade_cal.csv")
    cal_open = cal[(cal["exchange"] == "SSE") & (cal["is_open"] == 1)]
    cal_open = cal_open[(cal_open["cal_date"] >= 20160104) & (cal_open["cal_date"] <= 20260515)]
    expected_dates = set(cal_open["cal_date"])
    panel_dates = set(panel["trade_date"].unique())
    missing = expected_dates - panel_dates
    extra = panel_dates - expected_dates
    print(f"  expected trading days: {len(expected_dates)}")
    print(f"  panel trading days   : {len(panel_dates)}")
    print(f"  missing              : {len(missing)} {sorted(missing)[:5] if missing else ''}")
    print(f"  extra                : {len(extra)} {sorted(extra)[:5] if extra else ''}")


def check_nan_distribution():
    """[6] NaN分布看起来正常"""
    print("\n[6] NaN distribution (key cols)")
    panel = pd.read_parquet(CACHE / "panel.parquet")
    for col in ["close", "circ_mv", "pe_ttm", "net_mf_amount", "industry"]:
        na_pct = panel[col].isna().mean() * 100
        status = "OK"
        if col == "close" and na_pct > 0.1: status = "FAIL"
        if col == "circ_mv" and na_pct > 2: status = "FAIL"
        print(f"  {col:18s} na={na_pct:.2f}%  [{status}]")


if __name__ == "__main__":
    check_value_match()
    check_pct_chg()
    check_st_consistency()
    check_universe_content()
    check_date_continuity()
    check_nan_distribution()
    print("\n" + "=" * 60)
    print("All checks done.")
