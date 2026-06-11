"""
每日运营: 盘后一键运行 — 更新数据, 生成信号, 风控检查, 写日志

Usage:
  python -m code.live.daily_ops --date 20260601
  python -m code.live.daily_ops --date 20260601 --dry-run  # 只检查不执行
"""
import argparse, json, sys, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"; OUTPUT = ROOT / "output"
PORTFOLIO_DIR = OUTPUT / "portfolios"
SIGNAL_DIR = OUTPUT / "signals"
(OUTPUT / "monitor").mkdir(parents=True, exist_ok=True)

DATA_SOURCE = Path("E:/科大云盘/A股数据")


def step1_update_data(date):
    """Step 1: 增量更新数据"""
    print("\n" + "=" * 50)
    print("[1/5] 数据更新")
    daily_dir = DATA_SOURCE / "daily"
    target_file = daily_dir / f"{date}.csv"
    if not target_file.exists():
        print(f"  SKIP: {target_file} 不存在 (数据尚未上传)")
        return False
    from code.data.incremental_update import scan_new_dates, update_panel, update_factors, \
        update_labels, update_universe, update_features, update_market_features
    new_dates = scan_new_dates(CACHE / "panel.parquet", daily_dir)
    new_dates = [d for d in new_dates if d <= date]
    if not new_dates:
        print("  数据已是最新")
        return True
    print(f"  新日期: {new_dates}")
    update_panel(new_dates, DATA_SOURCE)
    update_factors(new_dates)
    update_labels(new_dates)
    update_universe(new_dates)
    update_features(new_dates)
    update_market_features(new_dates, DATA_SOURCE)
    print("  数据更新完成")
    return True


def step2_generate_signals(date, plans_config):
    """Step 2: 生成交易信号"""
    print("\n" + "=" * 50)
    print("[2/5] 信号生成")

    from code.live.daily_signal import predict_master
    feats = pd.read_parquet(CACHE / "features.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")

    all_signals = {}
    for plan_name, cfg in plans_config.items():
        model_type = cfg.get("model", "master")
        n = cfg.get("n", 8)

        if model_type == "master":
            day_feats = feats[feats["trade_date"] == date].copy()
            if len(day_feats) == 0:
                print(f"  {plan_name}: 无当日特征数据")
                continue
            codes, scores = predict_master(day_feats, feats, market, version="v1")
            if len(codes) == 0:
                print(f"  {plan_name}: 推理失败")
                continue
            score_map = dict(zip(codes, scores))
            basic = pd.read_csv(ROOT / "basic.csv")[["ts_code", "name"]]
            df = pd.DataFrame({"ts_code": list(score_map.keys()), "score": list(score_map.values())})
            df = df.sort_values("score", ascending=False)
            df = df.merge(basic, on="ts_code", how="left")
            top = df.head(n).copy()
            top["action"] = "buy"
            out_path = SIGNAL_DIR / f"{date}_{plan_name}.csv"
            top[["action", "ts_code", "name", "score"]].to_csv(out_path, index=False,
                                                                encoding="utf-8-sig")
            all_signals[plan_name] = top
            print(f"  {plan_name} ({model_type}, n={n}): {len(top)} stocks → {out_path}")

        elif model_type == "ensemble":
            from code.live.ensemble_signal import score_all
            df = score_all(date)
            if df is None:
                print(f"  {plan_name}: 集成推理失败")
                continue
            top = df.head(n)
            out_path = SIGNAL_DIR / f"{date}_{plan_name}.csv"
            top[["ts_code", "name", "ensemble_score"]].to_csv(out_path, index=False,
                                                               encoding="utf-8-sig")
            all_signals[plan_name] = top
            stock_list = ", ".join(f"{r['ts_code']} {r['name']}" for _, r in top.head(5).iterrows())
            print(f"  {plan_name} (ensemble, n={n}): {stock_list}")

    return all_signals


def step3_risk_check(date, plans_config):
    """Step 3: 风控检查"""
    print("\n" + "=" * 50)
    print("[3/5] 风控检查")

    from code.live.risk_manager import check_risk, get_risk_report, RiskState

    market = pd.read_parquet(CACHE / "market_features.parquet")
    mr = market[market["trade_date"] == date]
    vol_z = float(mr["hs300_vol_20"].values[0]) if len(mr) > 0 else 0.0

    # Load tracking for cumulative returns
    track_file = OUTPUT / "monitor" / "daily_track.json"
    track = json.load(open(track_file)) if track_file.exists() else {"plans": {}}

    results = {}
    for plan_name in plans_config:
        plan_key = f"plan_{plan_name}"
        tp = track["plans"].get(plan_key, {})
        nav = tp.get("nav", 1.0)
        cumulative_dd = nav / max(tp.get("peak_nav", 1.0), 1e-8) - 1
        weekly_ret = [r["ret"] for r in tp.get("daily_rets", [])][-5:]

        check = check_risk(plan_name, 0.0, cumulative_dd, vol_z, weekly_ret)
        results[plan_name] = check

        emoji = {"normal": "OK", "reduce": "WARN", "stop": "STOP", "paused": "PAUSE"}
        print(f"  方案{plan_name}: [{emoji.get(check.state.value, '?')}] {check.reason}")

    print(f"\n  风控状态:\n{get_risk_report()}")
    return results


def step4_log_and_position(date, all_signals, risk_results, plans_config):
    """Step 4: 记录日志 + 计算仓位"""
    print("\n" + "=" * 50)
    print("[4/5] 日志 & 仓位")

    from code.live.trade_journal import log_entry
    from code.live.position_size import main as calc_position
    import subprocess

    for plan_name, cfg in plans_config.items():
        risk = risk_results.get(plan_name)
        if risk and risk.state.value in ("stop", "paused"):
            print(f"  方案{plan_name}: 风控阻止交易 ({risk.state.value})")
            log_entry(date, plan_name, "stop", [], risk.reason)
            continue

        sig_path = SIGNAL_DIR / f"{date}_{plan_name}.csv"
        if not sig_path.exists():
            print(f"  方案{plan_name}: 无信号文件")
            continue

        sig = pd.read_csv(sig_path)
        stocks = [(r["ts_code"], r.get("name", ""), 0, 0) for _, r in sig.iterrows()]
        log_entry(date, plan_name, "buy" if cfg.get("init", True) else "rebalance",
                  stocks, f"模型选股, n={cfg.get('n',8)}",
                  {"model": cfg.get("model", "master"), "n": cfg.get("n", 8)})

        # Calculate positions
        capital = cfg["capital"]
        n = cfg.get("n", 8)
        cmd = [sys.executable, "-m", "code.live.position_size",
               "--signal", str(sig_path), "--capital", str(capital), "--top-n", str(n)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"\n  --- 方案{plan_name} (本金 ¥{capital:,}) ---")
        for line in result.stdout.split("\n")[-15:]:
            if line.strip():
                print(f"  {line}")


def step5_summary(date, all_signals, risk_results):
    """Step 5: 生成日报"""
    print("\n" + "=" * 50)
    print("[5/5] 日报")

    lines = [f"# 交易日报 — {date}", "",
             f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
             "## 信号", ""]

    for plan_name, sig in all_signals.items():
        lines.append(f"### 方案 {plan_name}")
        for _, r in sig.head(8).iterrows():
            lines.append(f"- {r['ts_code']} {r.get('name','')}: {r.get('score',r.get('ensemble_score',0)):.4f}")
        lines.append("")

    lines.append("## 风控")
    for plan_name, risk in risk_results.items():
        lines.append(f"- 方案{plan_name}: {risk.state.value} — {risk.reason}")

    report = "\n".join(lines)
    report_path = OUTPUT / "monitor" / f"report_{date}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  已保存: {report_path}")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 方案配置
    plans_config = {
        "A": {"capital": 1_000_000, "n": 8, "model": "master", "init": True},
        "B": {"capital": 60_000, "n": 5, "model": "ensemble", "init": True},
        "C": {"capital": 1_000, "n": 2, "model": "master", "init": True},
    }

    print(f"每日运营 — {args.date}")
    print(f"方案: A(v3单模型,100万,n=8) B(集成,6万,n=5) C(单模型,1千,n=2)")

    if args.dry_run:
        print("\n[DRY RUN] 仅检查, 不执行交易")

    # Step 1: 数据更新
    data_ok = step1_update_data(args.date) if not args.dry_run else True

    # Step 2: 信号
    all_signals = step2_generate_signals(args.date, plans_config)

    # Step 3: 风控
    risk_results = step3_risk_check(args.date, plans_config)

    # Step 4: 日志 + 仓位 (dry-run 跳过)
    if not args.dry_run and all_signals:
        step4_log_and_position(args.date, all_signals, risk_results, plans_config)

    # Step 5: 日报
    step5_summary(args.date, all_signals, risk_results)

    print(f"\n{'='*50}")
    print("OK 每日运营完成")


if __name__ == "__main__":
    main()
