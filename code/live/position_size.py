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
  次日换仓: 计算卖出回款 → 全部股票(含持有)等权重分配 → 调整仓位
  贪心模式 (--greedy): 按信号排名, 尽可能多买 #1, 余钱买 #2, 依次类推
"""
import argparse
import json
import pandas as pd
from pathlib import Path

from code.config import ROOT, CACHE, OUTPUT

STATE_DIR = OUTPUT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

FEE_SELL = 0.00125  # 0.025% 佣金 + 0.1% 印花税
FEE_BUY = 0.00025   # 0.025% 佣金


def save_holdings_state(date, portfolio, cash_left, plan="master"):
    """持久化当前持仓到 state 文件"""
    state = {
        "date": date,
        "plan": plan,
        "cash_left": round(cash_left, 2),
        "stocks": {code: int(lots) for code, lots in portfolio.items()},
    }
    path = STATE_DIR / "current_holdings.json"
    json.dump(state, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  持仓状态已保存: {path}")


def load_holdings_state():
    """读取上次持仓状态"""
    path = STATE_DIR / "current_holdings.json"
    if not path.exists():
        return None
    return json.load(open(path, encoding="utf-8"))


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


def _execute_sells(holdings_dict, sell_codes, price_map, name_map):
    """执行卖出, 返回 (sale_proceeds, kept_holdings)"""
    sale_proceeds = 0.0
    kept_holdings = {}  # {code: (lots, value)}
    for code, lots in holdings_dict.items():
        if code not in price_map:
            print(f"  WARN: {code} 无当日价格, 跳过")
            continue
        price = price_map[code]
        value = lots * price * 100
        if code in sell_codes and lots > 0:
            proceeds = value * (1 - FEE_SELL)
            sale_proceeds += proceeds
            print(f"  卖出 {code} {name_map.get(code, code)}: {lots}手×{price:.2f} 回款 {proceeds:,.0f}")
        else:
            kept_holdings[code] = (lots, value)
    return sale_proceeds, kept_holdings


def _adjust_held(kept_holdings, target_value, price_map, name_map):
    """调整持有股仓位, 返回 (transactions, portfolio, deployed, fee_drag)"""
    transactions, portfolio = [], {}
    deployed = 0.0
    fee_drag = 0.0
    for code, (cur_lots, cur_value) in kept_holdings.items():
        price = price_map[code]
        lot_val = price * 100
        name = name_map.get(code, code)
        delta = target_value - cur_value
        if delta >= lot_val:
            add_lots = int(delta / lot_val)
            new_lots = cur_lots + add_lots
            new_value = new_lots * lot_val
            transactions.append({"code": code, "name": name, "price": price,
                                 "lot_val": lot_val, "action": "加仓",
                                 "lots_before": cur_lots, "lots_after": new_lots,
                                 "actual": new_value})
            portfolio[code] = new_lots
            deployed += new_value
            print(f"  加仓 {code} {name}: {cur_lots}→{new_lots}手 (+{add_lots})")
        elif delta <= -lot_val and cur_lots > 1:
            sell_lots = min(cur_lots - 1, int(-delta / lot_val))
            if sell_lots > 0:
                new_lots = cur_lots - sell_lots
                new_value = new_lots * lot_val
                fee_drag += sell_lots * lot_val * FEE_SELL
                transactions.append({"code": code, "name": name, "price": price,
                                     "lot_val": lot_val, "action": "减仓",
                                     "lots_before": cur_lots, "lots_after": new_lots,
                                     "actual": new_value})
                portfolio[code] = new_lots
                deployed += new_value
                print(f"  减仓 {code} {name}: {cur_lots}→{new_lots}手 (-{sell_lots})")
            else:
                transactions.append({"code": code, "name": name, "price": price,
                                     "lot_val": lot_val, "action": "持有",
                                     "lots_before": cur_lots, "lots_after": cur_lots,
                                     "actual": cur_value})
                portfolio[code] = cur_lots
                deployed += cur_value
        else:
            transactions.append({"code": code, "name": name, "price": price,
                                 "lot_val": lot_val, "action": "持有",
                                 "lots_before": cur_lots, "lots_after": cur_lots,
                                 "actual": cur_value})
            portfolio[code] = cur_lots
            deployed += cur_value
    return transactions, portfolio, deployed, fee_drag


def _allocate_buys(buy_rows, target_value, total_pool, deployed, price_map, name_map):
    """为新买入分配资金, 返回 (transactions, portfolio, remaining_cash)"""
    buy_list = []
    for nr in buy_rows:
        lots = max(0, int(target_value / nr["lot_val"]))
        buy_list.append({**nr, "lots": lots, "actual": lots * nr["lot_val"]})

    initial_cost = sum(nb["actual"] for nb in buy_list)
    remaining = total_pool - deployed - initial_cost

    buy_list.sort(key=lambda x: x["lot_val"])
    for nb in buy_list:
        if remaining >= nb["lot_val"] and nb["lots"] > 0:
            nb["lots"] += 1
            nb["actual"] += nb["lot_val"]
            remaining -= nb["lot_val"]

    transactions, portfolio = [], {}
    for nb in buy_list:
        if nb["lots"] == 0:
            print(f"  ✗ {nb['code']} {nb['name']}: 1手需{nb['lot_val']:,.0f}, 买不起")
            continue
        transactions.append({"code": nb["code"], "name": nb["name"], "price": nb["price"],
                             "lot_val": nb["lot_val"], "action": "买入",
                             "lots_before": 0, "lots_after": nb["lots"],
                             "actual": nb["actual"]})
        portfolio[nb["code"]] = nb["lots"]
        deployed += nb["actual"]
    return transactions, portfolio, max(0, total_pool - deployed)


def full_rebalance(holdings_dict, sell_codes, buy_df, cash_left, day, basic):
    """
    完整等权调仓: 持有股也参与再平衡。

    Returns: (transactions, final_portfolio, remaining_cash)
    """
    price_map = {row["ts_code"]: float(row["close"]) for _, row in day.iterrows()}
    name_map = dict(zip(basic["ts_code"], basic["name"]))

    sale_proceeds, kept_holdings = _execute_sells(holdings_dict, sell_codes, price_map, name_map)
    kept_value = sum(v for _, (_, v) in kept_holdings.items())

    new_buy_rows = []
    for _, row in buy_df.iterrows():
        code = row["ts_code"]
        if code in price_map and code not in holdings_dict:
            new_buy_rows.append({"code": code,
                                 "name": row.get("name", name_map.get(code, "?")),
                                 "price": price_map[code],
                                 "lot_val": price_map[code] * 100})

    all_targets = list(kept_holdings.keys()) + [r["code"] for r in new_buy_rows]
    if not all_targets:
        print("\nERROR: 无目标股票")
        return [], {}, cash_left

    total_pool = sale_proceeds + kept_value + cash_left
    target_value = total_pool / len(all_targets)
    print(f"\n  总资产: {total_pool:,.0f} = 卖款{sale_proceeds:,.0f} + 持仓{kept_value:,.0f} + 现金{cash_left:,.0f}")
    print(f"  目标: {len(all_targets)}只, 每只 {target_value:,.0f} ({100/len(all_targets):.1f}%)")

    held_tx, held_pf, deployed, fee_drag = _adjust_held(kept_holdings, target_value, price_map, name_map)
    total_pool -= fee_drag

    buy_tx, buy_pf, remaining = _allocate_buys(new_buy_rows, target_value, total_pool, deployed, price_map, name_map)

    all_tx = held_tx + buy_tx
    all_pf = {**held_pf, **buy_pf}
    return all_tx, all_pf, remaining


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
    parser.add_argument("--plan", type=str, default="master",
                        help="方案名称 (用于持仓状态保存)")
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
    basic = pd.read_csv(ROOT / "basic.csv", usecols=["ts_code", "name"])

    # --top-n
    if args.top_n and args.top_n < len(buys_df):
        buys_df = buys_df.head(args.top_n)

    # ---- 换仓模式 ----
    holdings_dict = {}
    if args.holdings:
        for item in args.holdings.split(","):
            item = item.strip()
            if ":" in item:
                code, lots = item.split(":")
                holdings_dict[code.strip()] = int(lots)

    if not is_init and holdings_dict:
        # 使用完整调仓逻辑
        sell_codes = set(sells_df["ts_code"].tolist())
        transactions, final_portfolio, cash_remaining = full_rebalance(
            holdings_dict, sell_codes, buys_df, args.cash_left, day, basic)

        if not transactions:
            print("\n  无需调整")
            return

        # 显示结果
        total_actual = sum(t["actual"] for t in transactions)
        total_portfolio = total_actual + cash_remaining
        transactions.sort(key=lambda x: x["actual"], reverse=True)

        print(f"\n  最终持仓: {len(transactions)} 只\n")
        print(f"{'代码':<12s} {'名称':<8s} {'收盘价':>8s} {'手数(前→后)':>12s} {'金额':>12s} {'占比':>6s} {'操作':>4s}")
        print("-" * 80)
        for t in transactions:
            pct = t["actual"] / total_portfolio * 100 if total_portfolio > 0 else 0
            bar = "█" * int(pct / 2)
            lots_str = f"{t['lots_before']}→{t['lots_after']}"
            print(f"{t['code']:<12s} {t['name']:<8s} {t['price']:>8.2f} {lots_str:>12s} "
                  f"{t['actual']:>12,.0f} {pct:>5.1f}% {bar} {t['action']:>4s}")
        print("-" * 80)
        print(f"{'合计':>62} {total_actual:>12,.0f} {total_actual/total_portfolio*100:>5.1f}%")
        print(f"剩余现金: {cash_remaining:,.0f} 元 ({cash_remaining/total_portfolio*100:.1f}%)")

        # 保存持仓状态
        save_holdings_state(trade_date, final_portfolio, cash_remaining, args.plan)

    elif is_init or not holdings_dict:
        # 建仓模式 (首日或未传 holdings)
        buys = buys_df.merge(day[["ts_code", "close"]], on="ts_code", how="left")
        buys = buys.merge(basic, on="ts_code", how="left", suffixes=("_sig", ""))
        if "name" not in buys.columns:
            buys["name"] = buys.get("name_sig", "?")
        buys["name"] = buys["name"].fillna(buys.get("name_sig", "?"))
        buys["lot_value"] = buys["close"] * 100

        total = args.capital
        mode_str = "贪心" if args.greedy else "等权"
        n_buy = len(buys)
        print(f"日期: {trade_date}  资金: {total:,.0f}  模式: {mode_str}  买入: {n_buy}只 (建仓)")

        if args.skip_expensive and len(buys) > 0:
            affordable = []
            threshold = total if args.greedy else total / len(buys)
            for _, row in buys.iterrows():
                if row["lot_value"] <= threshold:
                    affordable.append(row)
                else:
                    print(f"  剔除 {row['ts_code']} {row['name']}: 1手需{row['lot_value']:,.0f} > {threshold:,.0f}")
            buys = pd.DataFrame(affordable)

        if len(buys) == 0:
            print("\nERROR: 没有买得起的股票")
            return

        buys = buys.reset_index(drop=True)
        if args.greedy:
            results, cash_left = greedy(buys, total)
        else:
            results, cash_left = equal_weight(buys, total)

        total_actual = sum(r["actual"] for r in results)
        print(f"\n  最终持仓: {len(results)} 只\n")
        print(f"{'代码':<12s} {'名称':<8s} {'收盘价':>8s} {'1手金额':>10s} {'建仓(手)':>8s} {'实际金额':>10s} {'占比':>6s}")
        print("-" * 72)
        for r in results:
            pct = r["actual"] / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"{r['code']:<12s} {r['name']:<8s} {r['price']:>8.2f} {r['lot_val']:>10,.0f} "
                  f"{r['lots']:>8d} {r['actual']:>10,.0f} {pct:>5.1f}% {bar}")
        print("-" * 72)
        print(f"{'合计':<12s} {'':<8s} {'':>8s} {'':>10s} {'':>8s} {total_actual:>10,.0f} {total_actual/total*100:>5.1f}%")
        print(f"剩余现金: {cash_left:,.0f} 元 ({cash_left/total*100:.1f}%)")

        # 保存持仓状态
        portfolio = {r["code"]: r["lots"] for r in results}
        save_holdings_state(trade_date, portfolio, cash_left, args.plan)


if __name__ == "__main__":
    main()
