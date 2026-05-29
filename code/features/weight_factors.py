"""
指数权重因子: 读取 index_weight/ 月度成分股权重, 生成日频因子。

输出 cache/weight_factors.parquet:
  trade_date, ts_code, hs300_weight, hs300_dweight, cyb_weight

逻辑:
  - 月度权重前向填充到每日 (按交易日历)
  - hs300_dweight: 仅权重变动日 = 新权重 - 旧权重, 其余日 = 0
  - 不在指数内的股票 weight = 0
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)


def main():
    idx_dir = ROOT / "index_weight"
    trade_cal = pd.read_csv(ROOT / "trade_cal.csv")
    cal_dates = sorted(trade_cal["cal_date"].astype(int).unique())

    # ---- Load HS300 weights ----
    hs300_files = sorted(idx_dir.glob("*_000300.SH.csv"))
    print(f"HS300 weight files: {len(hs300_files)}")

    hs300_parts = []
    for f in tqdm(hs300_files, desc="HS300"):
        df = pd.read_csv(f)
        df = df.rename(columns={"con_code": "ts_code"})
        hs300_parts.append(df[["ts_code", "trade_date", "weight"]])
    hs300 = pd.concat(hs300_parts, ignore_index=True)
    hs300 = hs300.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    hs300["trade_date"] = hs300["trade_date"].astype(int)
    hs300 = hs300.rename(columns={"weight": "hs300_weight"})
    print(f"  HS300 raw rows: {len(hs300):,}")

    # ---- Load ChiNext weights ----
    cyb_files = sorted(idx_dir.glob("*_399006.SZ.csv"))
    print(f"CYB weight files: {len(cyb_files)}")

    cyb_parts = []
    for f in tqdm(cyb_files, desc="CYB"):
        df = pd.read_csv(f)
        df = df.rename(columns={"con_code": "ts_code"})
        cyb_parts.append(df[["ts_code", "trade_date", "weight"]])
    cyb = pd.concat(cyb_parts, ignore_index=True)
    cyb = cyb.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    cyb["trade_date"] = cyb["trade_date"].astype(int)
    cyb = cyb.rename(columns={"weight": "cyb_weight"})
    print(f"  CYB raw rows: {len(cyb):,}")

    # ---- Forward-fill to daily ----
    # 每月月末日期的权重, 次交易日生效, 持续到下次更新
    # 简化: 按 ts_code group, 在每个权重变动日填充, 然后 reindex 到 cal_dates 并 ffill

    def forward_fill_monthly(monthly_df, weight_col):
        """将月度权重前向填充到每日"""
        df = monthly_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        # 获取每只股票的所有权重变动日期
        result_rows = []
        for code, grp in tqdm(df.groupby("ts_code"), desc=f"ffill {weight_col}", leave=False):
            grp = grp.sort_values("trade_date")
            # 为每个变动日期确定下一交易日作为生效日
            # 简化: 直接使用 trade_date 作为生效日期, ffill 到后续日历日
            date_weight = dict(zip(grp["trade_date"], grp[weight_col]))

            # 找到股票的起始日期 (第一个有权重的日期)
            first_date = grp["trade_date"].iloc[0]
            last_date = grp["trade_date"].iloc[-1]

            # 对每个日历日, 找到 <= 该日的最近权重
            current_weight = 0.0
            weight_dates = sorted(date_weight.keys())
            wi = 0

            stock_rows = []
            for d in cal_dates:
                if d < first_date:
                    stock_rows.append((code, d, 0.0))
                    continue
                # 推进到最新的权重变动
                while wi < len(weight_dates) and weight_dates[wi] <= d:
                    current_weight = date_weight[weight_dates[wi]]
                    wi += 1
                stock_rows.append((code, d, current_weight))

            result_rows.extend(stock_rows)

        result = pd.DataFrame(result_rows, columns=["ts_code", "trade_date", weight_col])
        return result

    hs300_daily = forward_fill_monthly(hs300, "hs300_weight")
    cyb_daily = forward_fill_monthly(cyb, "cyb_weight")

    # ---- Merge ----
    merged = hs300_daily.merge(cyb_daily, on=["ts_code", "trade_date"], how="outer")
    merged["hs300_weight"] = merged["hs300_weight"].fillna(0.0).astype("float32")
    merged["cyb_weight"] = merged["cyb_weight"].fillna(0.0).astype("float32")

    # ---- Compute hs300_dweight ----
    merged = merged.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    merged["hs300_dweight"] = (
        merged.groupby("ts_code")["hs300_weight"]
        .diff()
        .fillna(0.0)
        .astype("float32")
    )

    # 统计
    nonzero_dw = (merged["hs300_dweight"] != 0).sum()
    nonzero_w = (merged["hs300_weight"] > 0).sum()
    nonzero_cyb = (merged["cyb_weight"] > 0).sum()
    print(f"\n  hs300_weight > 0: {nonzero_w:,} rows ({nonzero_w/len(merged)*100:.1f}%)")
    print(f"  hs300_dweight != 0: {nonzero_dw:,} rows")
    print(f"  cyb_weight > 0: {nonzero_cyb:,} rows ({nonzero_cyb/len(merged)*100:.1f}%)")
    print(f"  Total rows: {len(merged):,}")

    # ---- Save ----
    out_path = CACHE / "weight_factors.parquet"
    merged.to_parquet(out_path, index=False, compression="snappy")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
