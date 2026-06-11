"""
实盘监控: 每日盘后运行, 跟踪三方案收益

Usage:
  python -m code.live.daily_monitor --date 20260601
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"; CACHE = ROOT / "cache"
PORTFOLIO_DIR = OUTPUT / "portfolios"
(OUTPUT / "monitor").mkdir(parents=True, exist_ok=True)


def load_portfolio(date):
    """加载最近的持仓记录"""
    for offset in range(5):
        d = date - offset
        f = PORTFOLIO_DIR / f"{d}_initial.json"
        if f.exists():
            return int(d), json.load(open(f))
    return None, None


def fetch_today_prices(date, codes):
    """获取当日收盘价"""
    panel = pd.read_parquet(CACHE / "panel.parquet",
        columns=["trade_date","ts_code","close","pct_chg"])
    day = panel[panel["trade_date"]==date]
    prices = {}
    for code in codes:
        row = day[day["ts_code"]==code]
        if len(row) > 0:
            prices[code] = {"close": float(row["close"].values[0]),
                           "pct_chg": float(row["pct_chg"].values[0])}
    return prices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=int, required=True)
    args = parser.parse_args()

    today = args.date
    port_date, portfolio = load_portfolio(today)
    if portfolio is None:
        print(f"No portfolio found near {today}")
        return

    print(f"持仓日期: {port_date} → 交易日期: {today}")
    print(f"{'='*70}")

    # HS300 benchmark
    hs300 = pd.read_csv(ROOT / "market" / "000300.SH.csv")
    hs300 = hs300[hs300["trade_date"]==today]
    hs300_ret = float(hs300["pct_chg"].values[0]) / 100 if len(hs300) > 0 else 0.0

    # Initialize tracking if first day
    track_file = OUTPUT / "monitor" / "daily_track.json"
    if track_file.exists():
        track = json.load(open(track_file))
    else:
        track = {"days": [], "plans": {}}

    for plan_name, plan in portfolio["plans"].items():
        codes = list(plan["holdings"].keys())
        prices = fetch_today_prices(today, codes)

        # Calculate daily return
        daily_ret = 0.0
        total_value = 0.0
        stock_rets = {}

        for code, holding in plan["holdings"].items():
            lots = holding["lots"]
            close_yesterday = holding["close"]
            if code in prices:
                close_today = prices[code]["close"]
                pct = prices[code]["pct_chg"] / 100
                value = lots * 100 * close_today
                total_value += value
                stock_rets[code] = {"pct": pct, "value": value}
            else:
                # Stock not in panel (halted/suspended) — use yesterday's price
                value = lots * 100 * close_yesterday
                total_value += value
                stock_rets[code] = {"pct": 0.0, "value": value, "halted": True}

        # Weighted return
        for code, sr in stock_rets.items():
            daily_ret += sr["pct"] * (sr["value"] / total_value) if total_value > 0 else 0

        # Track
        plan_key = f"plan_{plan_name}"
        if plan_key not in track["plans"]:
            track["plans"][plan_key] = {
                "capital": plan["capital"],
                "nav": 1.0,
                "daily_rets": [],
                "cash_left": plan["cash_left"],
            }

        tp = track["plans"][plan_key]
        tp["nav"] *= (1 + daily_ret)
        tp["daily_rets"].append({"date": today, "ret": daily_ret, "nav": tp["nav"]})

        invested = plan["total_invested"]
        cash = plan["cash_left"]
        total_asset = invested * (1 + daily_ret) + cash

        print(f"\n方案 {plan_name} (本金 ¥{plan['capital']:,})")
        print(f"  日收益: {daily_ret*100:+.2f}%  累计净值: {tp['nav']:.4f}  总资产: ¥{total_asset:,.0f}")
        print(f"  HS300:  {hs300_ret*100:+.2f}%  超额: {(daily_ret-hs300_ret)*100:+.2f}%")

        for code in list(stock_rets.keys())[:5]:
            sr = stock_rets[code]
            halted = sr.get("halted", False)
            print(f"    {code} {holding['name'] if code in plan['holdings'] else ''}: {sr['pct']*100:+.2f}% {'[停牌]' if halted else ''}")

    print(f"\nHS300: {hs300_ret*100:+.2f}%")

    # Save tracking
    track["days"].append(today)
    json.dump(track, open(track_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {track_file}")


if __name__ == "__main__":
    main()
