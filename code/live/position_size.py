"""
仓位计算工具: 根据信号文件 + 当日收盘价, 计算每只股票应买多少手。

Usage:
  python -m code.live.position_size --signal output/signals/20260521_master.csv \\
      --capital 1000000
  python -m code.live.position_size --signal output/signals/20260521_master.csv \\
      --capital 1000000 --skip-expensive

逻辑:
  1. 读取信号文件中的 buy 清单
  2. 获取当日收盘价
  3. 等权分配: 总资金 / N → 每只目标金额
  4. 目标金额 / (股价 × 100) → 手数 (向下取整)
  5. --skip-expensive: 剔除买不起1手的, 剩余等权重分配
"""
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"


def main():
    parser = argparse.ArgumentParser(description="仓位计算")
    parser.add_argument("--signal", type=str, required=True, help="信号 CSV 路径")
    parser.add_argument("--capital", type=float, required=True, help="总资金 (元)")
    parser.add_argument("--skip-expensive", action="store_true",
                        help="剔除买不起1手的股票, 剩余等权重分配")
    args = parser.parse_args()

    signal_path = Path(args.signal)
    if not signal_path.exists():
        print(f"ERROR: signal file not found: {signal_path}")
        return

    df = pd.read_csv(signal_path)
    buys = df[df["action"] == "buy"].copy()
    if buys.empty:
        buys = df.copy()  # 全是 buy (建仓)

    # 获取当日收盘价
    date_str = signal_path.stem.split("_")[0]
    trade_date = int(date_str)
    panel = pd.read_parquet(CACHE / "panel.parquet", columns=["trade_date", "ts_code", "close"])
    day = panel[panel["trade_date"] == trade_date]
    basic = pd.read_csv(ROOT / "basic.csv")[["ts_code", "name"]]

    # 合并股价和名称 (信号文件自带 name, 用 basic 覆盖保证准确, 缺失用信号原值回退)
    buys = buys.merge(day[["ts_code", "close"]], on="ts_code", how="left")
    buys = buys.merge(basic, on="ts_code", how="left", suffixes=("_sig", ""))
    if "name" not in buys.columns:
        buys["name"] = buys.get("name_sig", "?")
    buys["name"] = buys["name"].fillna(buys.get("name_sig", "?"))
    buys["lot_value"] = buys["close"] * 100  # 1手金额

    total = args.capital
    n_original = len(buys)

    print(f"日期: {trade_date}  总资金: {total:,.0f}  信号数: {n_original}")

    # --skip-expensive: 剔除买不起1手的
    skipped = []
    if args.skip_expensive:
        affordable = []
        for _, row in buys.iterrows():
            if row["lot_value"] <= total / (n_original - len(skipped)):
                affordable.append(row)
            else:
                skipped.append(row)
        if skipped:
            print(f"\n  剔除 {len(skipped)} 只高价股:")
            for s in skipped:
                print(f"    - {s['ts_code']} {s['name']} 收盘{s['close']:.2f} 1手需{s['lot_value']:,.0f}")
            # 用剩余可负担股票重建
            buys = pd.DataFrame(affordable)
    else:
        # 不过滤, 但标记买不起的为 0 手
        pass

    buys = buys.reset_index(drop=True)
    n = len(buys)
    equal = total / n

    print(f"  最终持仓: {n} 只  等权: {equal:,.0f}/只\n")
    print(f"{'代码':<12s} {'名称':<8s} {'收盘价':>8s} {'1手金额':>10s} {'建仓(手)':>8s} {'实际金额':>10s} {'占比':>6s}")
    print("-" * 72)

    total_actual = 0
    for _, row in buys.iterrows():
        code = row["ts_code"]
        name = row["name"]
        price = float(row["close"])
        lot_val = float(row["lot_value"])
        lots = max(0, int(equal / lot_val))
        actual = lots * lot_val
        total_actual += actual
        pct = actual / total * 100
        bar = "█" * int(pct / 2)
        print(f"{code:<12s} {name:<8s} {price:>8.2f} {lot_val:>10,.0f} {lots:>8d} {actual:>10,.0f} {pct:>5.1f}% {bar}")

    print("-" * 72)
    print(f"{'合计':<12s} {'':<8s} {'':>8s} {'':>10s} {'':>8s} {total_actual:>10,.0f} {total_actual/total*100:>5.1f}%")
    cash_left = total - total_actual
    print(f"剩余现金: {cash_left:,.0f} 元 ({cash_left/total*100:.1f}%)")


if __name__ == "__main__":
    main()
