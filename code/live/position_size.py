"""
仓位计算工具: 根据信号文件 + 当日收盘价, 计算每只股票应买多少手。

Usage:
  # 大作业模式 (100万, 10只等权)
  python -m code.live.position_size --signal output/signals/20260521_master.csv \
      --capital 1000000 --skip-expensive

  # 换仓模式 (次日, 传当前持仓 + 剩余现金)
  python -m code.live.position_size --signal output/signals/20260602_master.csv \
      --capital 1000000 --skip-expensive \
      --holdings "601988.SH:192,601288.SH:179,600015.SH:170" --cash-left 23000

  # 私人模式 (1000元, 贪心分配, 只买前3名)
  python -m code.live.position_size --signal output/signals/20260521_master.csv \
      --capital 1000 --top-n 3 --skip-expensive --greedy

逻辑:
  首日建仓: 总资金 / N → 每只目标金额 → 向下取整手 → 余钱再分配
  次日换仓: 计算卖出回款 → 可用 = 剩余现金 + 卖款 → 等权分配买入
  贪心模式 (--greedy): 按信号排名, 尽可能多买 #1, 余钱买 #2, 依次类推
"""
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"

FEE_SELL = 0.00125  # 0.025% 佣金 + 0.1% 印花税


def equal_weight(buys, total):
    """等权分配 + 余钱再分配"""
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
    """贪心分配: 尽可能多买 #1, 余钱买 #2 ..."""
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
    parser.add_argument("--holdings", type=str, default="",
                        help="当前持仓 code:lots,... (换仓模式)")
    parser.add_argument("--cash-left", type=float, default=0,
                        help="上日剩余现金 (换仓模式, 配合 --holdings)")
    args = parser.parse_args()

    signal_path = Path(args.signal)
    if not signal_path.exists():
        print(f"ERROR: signal file not found: {signal_path}")
        return

    df = pd.read_csv(signal_path)
    sells_df = df[df["action"] == "sell"].copy()
    buys_df = df[df["action"] == "buy"].copy()

    is_init = sells_df.empty and buys_df.empty
    if is_init:
        buys_df = df.copy()

    # 获取当日收盘价
    date_str = signal_path.stem.split("_")[0]
    trade_date = int(date_str)
    panel = pd.read_parquet(CACHE / "panel.parquet", columns=["trade_date", "ts_code", "close"])
    day = panel[panel["trade_date"] == trade_date]
    basic = pd.read_csv(ROOT / "basic.csv")[["ts_code", "name"]]

    # --top-n
    if args.top_n and args.top_n < len(buys_df):
        buys_df = buys_df.head(args.top_n)

    # ---- 换仓模式: 计算可用资金 ----
    sale_proceeds = 0.0
    holdings_dict = {}
    holdings_str = ""
    if args.holdings and not is_init:
        for item in args.holdings.split(","):
            item = item.strip()
            if ":" in item:
                code, lots = item.split(":")
                holdings_dict[code.strip()] = int(lots)

        # 卖出回款
        sold_codes = set(sells_df["ts_code"].tolist())
        kept_value = 0
        for code, lots in holdings_dict.items():
            row = day[day["ts_code"] == code]
            if len(row) == 0:
                print(f"  WARN: {code} 无当日价格, 跳过")
                continue
            price = float(row["close"].iloc[0])
            name_val = basic[basic["ts_code"] == code]["name"].iloc[0] if len(basic[basic["ts_code"] == code]) > 0 else code
            if code in sold_codes and lots > 0:
                proceeds = lots * price * 100
                fee = proceeds * FEE_SELL
                sale_proceeds += proceeds - fee
                print(f"  卖出 {code} {name_val}: {lots}手×{price:.2f} 回款 {proceeds-fee:,.0f} (扣费{fee:,.0f})")
            else:
                kept_value += lots * price * 100

        # 只买新股票 (不在当前持仓中的)
        buys_df = buys_df[~buys_df["ts_code"].isin(holdings_dict.keys())].copy()
        if len(buys_df) == 0:
            print("\n  所有买入信号已在持仓中, 无需调整")
            return

        holdings_str = f"  持仓 {len(holdings_dict)}只, 卖出 {len(sold_codes)}只, 新买 {len(buys_df)}只"

    # 合并股价
    buys = buys_df.merge(day[["ts_code", "close"]], on="ts_code", how="left")
    buys = buys.merge(basic, on="ts_code", how="left", suffixes=("_sig", ""))
    if "name" not in buys.columns:
        buys["name"] = buys.get("name_sig", "?")
    buys["name"] = buys["name"].fillna(buys.get("name_sig", "?"))
    buys["lot_value"] = buys["close"] * 100

    total = args.capital
    if sale_proceeds > 0:
        available = sale_proceeds + args.cash_left
        total = available
        print(f"\n  卖出回款: {sale_proceeds:,.0f}  剩余现金: {args.cash_left:,.0f}  可用: {available:,.0f}")
    elif not is_init and not args.holdings:
        print("\n  ⚠ 换仓但未传 --holdings, 将使用完整 --capital 计算 (应为建仓模式)")

    mode_str = "贪心" if args.greedy else "等权"
    print(f"日期: {trade_date}  可用资金: {total:,.0f}  模式: {mode_str}  买入: {len(buys)}只{holdings_str}")

    # --skip-expensive
    skipped = []
    if args.skip_expensive and len(buys) > 0:
        affordable = []
        threshold = total if args.greedy else total / len(buys)
        for _, row in buys.iterrows():
            if row["lot_value"] <= threshold:
                affordable.append(row)
            else:
                skipped.append(row)
        if skipped:
            reason = "总资金" if args.greedy else f"等权{total/len(buys):,.0f}"
            print(f"\n  剔除 {len(skipped)} 只高价股 (1手>{reason}):")
            for s in skipped:
                print(f"    - {s['ts_code']} {s['name']} 收盘{s['close']:.2f} 1手需{s['lot_value']:,.0f}")
        buys = pd.DataFrame(affordable)

    if len(buys) == 0:
        print("\nERROR: 没有买得起的股票")
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
        pct = r["actual"] / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{r['code']:<12s} {r['name']:<8s} {r['price']:>8.2f} {r['lot_val']:>10,.0f} "
              f"{r['lots']:>8d} {r['actual']:>10,.0f} {pct:>5.1f}% {bar}")

    print("-" * 72)
    print(f"{'合计':<12s} {'':<8s} {'':>8s} {'':>10s} {'':>8s} {total_actual:>10,.0f} {total_actual/total*100:>5.1f}%" if total > 0 else "")
    cash_left_final = total - total_actual + (args.cash_left if sale_proceeds > 0 and cash_left > 0 else cash_left)
    if sale_proceeds > 0:
        print(f"剩余现金: {cash_left:,.0f} 元 (下日可用)")
    else:
        print(f"剩余现金: {cash_left:,.0f} 元 ({cash_left/total*100:.1f}%)" if total > 0 else f"剩余现金: {cash_left:,.0f} 元")


if __name__ == "__main__":
    main()
