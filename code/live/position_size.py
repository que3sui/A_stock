"""
仓位计算工具: 根据信号文件 + 当日收盘价, 计算每只股票应买多少手。

Usage:
  # 大作业模式 (100万, 10只等权)
  python -m code.live.position_size --signal output/signals/20260521_master.csv \\
      --capital 1000000 --skip-expensive

  # 私人模式 (1000元, 贪心分配, 只买前3名)
  python -m code.live.position_size --signal output/signals/20260521_master.csv \\
      --capital 1000 --top-n 3 --skip-expensive --greedy

逻辑:
  等权模式 (默认): 总资金 / N → 每只目标金额 → 向下取整手 → 余钱再分配
  贪心模式 (--greedy): 按信号排名, 尽可能多买 #1, 余钱买 #2, 依次类推
"""
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"


def equal_weight(buys, total):
    """等权分配 + 余钱再分配 (适合大资金多股票)"""
    n = len(buys)
    equal = total / n

    results = []
    total_actual = 0
    for _, row in buys.iterrows():
        lot_val = float(row["lot_value"])
        lots = max(0, int(equal / lot_val))
        actual = lots * lot_val
        total_actual += actual
        results.append({"code": row["ts_code"], "name": row["name"],
                       "price": float(row["close"]), "lot_val": lot_val,
                       "lots": lots, "actual": actual})

    # 余钱再分配: 按 lot_val 从小到大, 给已有持仓各加 1 手
    cash_left = total - total_actual
    results.sort(key=lambda x: x["lot_val"])
    for r in results:
        if cash_left >= r["lot_val"] and r["lots"] > 0:
            r["lots"] += 1
            r["actual"] += r["lot_val"]
            total_actual += r["lot_val"]
            cash_left = total - total_actual

    return results, cash_left


def greedy(buys, total):
    """贪心分配: 尽可能多买 #1, 余钱买 #2 ... (适合小资金)"""
    results = []
    cash = total

    for _, row in buys.iterrows():
        if cash <= 0:
            break
        lot_val = float(row["lot_value"])
        max_lots = int(cash / lot_val)
        if max_lots == 0:
            continue
        actual = max_lots * lot_val
        cash -= actual
        results.append({"code": row["ts_code"], "name": row["name"],
                       "price": float(row["close"]), "lot_val": lot_val,
                       "lots": max_lots, "actual": actual})

    return results, cash


def main():
    parser = argparse.ArgumentParser(description="仓位计算")
    parser.add_argument("--signal", type=str, required=True, help="信号 CSV 路径")
    parser.add_argument("--capital", type=float, required=True, help="总资金 (元)")
    parser.add_argument("--top-n", type=int, default=0,
                        help="只取信号前 N 名 (0=全部)")
    parser.add_argument("--skip-expensive", action="store_true",
                        help="剔除买不起1手的股票")
    parser.add_argument("--greedy", action="store_true",
                        help="贪心分配: 按排名依次满仓 (适合小资金)")
    args = parser.parse_args()

    signal_path = Path(args.signal)
    if not signal_path.exists():
        print(f"ERROR: signal file not found: {signal_path}")
        return

    df = pd.read_csv(signal_path)
    buys = df[df["action"] == "buy"].copy()
    if buys.empty:
        buys = df.copy()  # 全是 buy (建仓)

    # --top-n: 只取前 N 名
    if args.top_n and args.top_n < len(buys):
        buys = buys.head(args.top_n)

    # 获取当日收盘价
    date_str = signal_path.stem.split("_")[0]
    trade_date = int(date_str)
    panel = pd.read_parquet(CACHE / "panel.parquet", columns=["trade_date", "ts_code", "close"])
    day = panel[panel["trade_date"] == trade_date]
    basic = pd.read_csv(ROOT / "basic.csv")[["ts_code", "name"]]

    buys = buys.merge(day[["ts_code", "close"]], on="ts_code", how="left")
    buys = buys.merge(basic, on="ts_code", how="left", suffixes=("_sig", ""))
    if "name" not in buys.columns:
        buys["name"] = buys.get("name_sig", "?")
    buys["name"] = buys["name"].fillna(buys.get("name_sig", "?"))
    buys["lot_value"] = buys["close"] * 100

    total = args.capital
    mode_str = "贪心" if args.greedy else "等权"
    print(f"日期: {trade_date}  总资金: {total:,.0f}  模式: {mode_str}  信号数: {len(buys)}")

    # --skip-expensive: 剔除买不起1手的
    skipped = []
    if args.skip_expensive:
        affordable = []
        for _, row in buys.iterrows():
            if row["lot_value"] <= total:
                affordable.append(row)
            else:
                skipped.append(row)
        if skipped:
            print(f"\n  剔除 {len(skipped)} 只高价股 (>总资金):")
            for s in skipped:
                print(f"    - {s['ts_code']} {s['name']} 收盘{s['close']:.2f} 1手需{s['lot_value']:,.0f}")
        buys = pd.DataFrame(affordable)

    if len(buys) == 0:
        print("\nERROR: 没有买得起的股票, 尝试降低 --top-n 或增加资金")
        return

    buys = buys.reset_index(drop=True)

    # 分配
    if args.greedy:
        results, cash_left = greedy(buys, total)
    else:
        results, cash_left = equal_weight(buys, total)

    total_actual = total - cash_left
    n = len(results)

    print(f"\n  最终持仓: {n} 只\n")
    print(f"{'代码':<12s} {'名称':<8s} {'收盘价':>8s} {'1手金额':>10s} {'建仓(手)':>8s} {'实际金额':>10s} {'占比':>6s}")
    print("-" * 72)

    for r in results:
        pct = r["actual"] / total * 100
        bar = "█" * int(pct / 2)
        print(f"{r['code']:<12s} {r['name']:<8s} {r['price']:>8.2f} {r['lot_val']:>10,.0f} "
              f"{r['lots']:>8d} {r['actual']:>10,.0f} {pct:>5.1f}% {bar}")

    print("-" * 72)
    print(f"{'合计':<12s} {'':<8s} {'':>8s} {'':>10s} {'':>8s} {total_actual:>10,.0f} {total_actual/total*100:>5.1f}%")
    print(f"剩余现金: {cash_left:,.0f} 元 ({cash_left/total*100:.1f}%)")


if __name__ == "__main__":
    main()
