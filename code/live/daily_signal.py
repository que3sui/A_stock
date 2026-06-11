"""
模拟交易当日信号生成
  - 在 t 日盘后运行: 加载最新 panel, 推理模型, 输出 t+1 日的买/卖清单
  - 输出 output/signals/{date}_{model}.csv : (action, ts_code, name, score)

Usage:
  # 用 LightGBM:
  python -m code.live.daily_signal --date 20260601 --model lgbm  --n 10 --k 2
  # 用 GRU:
  python -m code.live.daily_signal --date 20260601 --model gru   --n 10 --k 2
  # 用 MASTER (v1):
  python -m code.live.daily_signal --date 20260601 --model master --n 10 --k 2
  # 用 MASTER v2:
  python -m code.live.daily_signal --date 20260601 --model master_v2 --n 10 --k 2
  # 用三模型 ensemble (需要先跑过各自的训练):
  python -m code.live.daily_signal --date 20260601 --model ensemble --n 10 --k 2

  --portfolio: 当前持仓的 ts_code 清单 (用逗号分隔), 用于决定卖出
                未提供则建仓 (买入 top n)
"""
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from code.config import ROOT, CACHE, OUTPUT, FACTOR_COLS, WEIGHT_COLS

(OUTPUT / "signals").mkdir(parents=True, exist_ok=True)


def predict_lgbm(features_df):
    """加载 lgbm 模型, 对当日 universe 内股票预测 score"""
    with open(OUTPUT / "checkpoints" / "lgbm.pkl", "rb") as f:
        model = pickle.load(f)
    X = features_df[FACTOR_COLS].fillna(0).values.astype(np.float32)
    scores = model.predict(X)
    return features_df["ts_code"].values, scores


def _gather_history(target_codes, target_date, all_features_df, T):
    """对每只股票, 取 trade_date <= target_date 的最近 T 天 features, 缺失用 0 填.
    Returns: (valid_codes, X, X_w) — X_w 为权重通道, 若列缺失则为 None"""
    hist = all_features_df[all_features_df["trade_date"] <= target_date]
    hist = hist[hist["ts_code"].isin(target_codes)]
    hist = hist.sort_values(["ts_code", "trade_date"])

    weight_cols = [c for c in WEIGHT_COLS if c in hist.columns]
    cols = FACTOR_COLS + weight_cols

    g = hist.groupby("ts_code", sort=False)
    X_list, valid_codes = [], []
    for code in target_codes:
        if code in g.indices:
            grp = hist.iloc[g.indices[code]]
            if len(grp) >= T:
                tail = grp.tail(T)[cols].fillna(0).values.astype(np.float32)
                X_list.append(tail)
                valid_codes.append(code)
    if not X_list:
        return np.array([]), np.array([], dtype=np.float32), None
    X_all = np.stack(X_list)
    n_feat = len(FACTOR_COLS)
    X = X_all[:, :, :n_feat]
    X_w = X_all[:, :, n_feat:] if weight_cols else None
    return np.array(valid_codes), X, X_w


def predict_gru(features_df, all_features_df, T=20):
    """加载 gru 模型, 用过去 T 天 features 推理"""
    from code.models.gru_att import GRUAtt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRUAtt(n_feat=len(FACTOR_COLS), hidden=64, num_layers=2, dropout=0.4).to(device)
    model.load_state_dict(torch.load(OUTPUT / "checkpoints" / "gru_att.pt",
                                     map_location=device, weights_only=True))
    model.eval()

    target_date = int(features_df["trade_date"].iloc[0])
    codes, X, _ = _gather_history(features_df["ts_code"].values, target_date, all_features_df, T)
    if len(codes) == 0:
        return np.array([]), np.array([])

    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        scores = model(X_t).cpu().numpy()
    return codes, scores


def predict_master_multiseed(features_df, all_features_df, market_df, seeds=(42, 1337, 2024)):
    """MASTER v3: 加载多 seed 模型, 推理后 rank-percentile 平均"""
    target_date = int(features_df["trade_date"].iloc[0])
    all_scores = []
    valid_codes_ref = None
    for s in seeds:
        ckpt_name = "master.pt" if s == 42 else f"master_seed{s}.pt"
        ckpt_path = OUTPUT / "checkpoints" / ckpt_name
        if not ckpt_path.exists():
            print(f"  WARN: {ckpt_path} not found, skip seed {s}")
            continue
        # 临时切换 ckpt 路径
        codes, scores = _predict_master_with_ckpt(
            features_df, all_features_df, market_df, ckpt_path
        )
        if len(codes) == 0:
            continue
        if valid_codes_ref is None:
            valid_codes_ref = codes
        s_series = pd.Series(scores, index=codes)
        s_series = s_series.rank(pct=True)
        all_scores.append(s_series)
    if not all_scores:
        return np.array([]), np.array([])
    df = pd.concat(all_scores, axis=1).dropna(how="all")
    df = df.fillna(df.mean())
    avg_score = df.mean(axis=1)
    return avg_score.index.values, avg_score.values


def _predict_master_with_ckpt(features_df, all_features_df, market_df, ckpt_path):
    """通用 master 推理 (任意 v1 风格 ckpt)"""
    from code.models.master import MASTER, T as MASTER_T, N_WEIGHT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    F_market = len([c for c in market_df.columns if c != "trade_date"])
    F_weight = N_WEIGHT if any(c in all_features_df.columns for c in WEIGHT_COLS) else 0
    model = MASTER(F_stock=len(FACTOR_COLS), F_market=F_market, H=64, T=MASTER_T,
                   nhead=4, dropout=0.2, n_intra_layers=2, n_inter_layers=1,
                   F_weight=F_weight).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    target_date = int(features_df["trade_date"].iloc[0])
    codes, X, X_w = _gather_history(features_df["ts_code"].values, target_date, all_features_df, MASTER_T)
    if len(codes) == 0:
        return np.array([]), np.array([])

    market_df = market_df.sort_values("trade_date").reset_index(drop=True)
    market_cols = [c for c in market_df.columns if c != "trade_date"]
    market_X_all = market_df[market_cols].fillna(0).values.astype(np.float32)
    market_dates = market_df["trade_date"].values
    mask = market_dates <= target_date
    if mask.sum() < MASTER_T:
        return np.array([]), np.array([])
    m_end = int(np.where(mask)[0][-1])
    market_window = market_X_all[m_end - MASTER_T + 1 : m_end + 1]

    X_t = torch.from_numpy(X).to(device)
    m_t = torch.from_numpy(market_window).to(device)
    X_w_t = torch.from_numpy(X_w).to(device) if X_w is not None else None
    with torch.no_grad():
        scores = model(X_t, m_t, X_w_t).cpu().numpy()
    return codes, scores


def predict_master(features_df, all_features_df, market_df, version="v1"):
    """
    MASTER 推理 (v1 或 v2)
      - 加载 master 模型 + market features
      - 用每只股票过去 T 天 features + 整体过去 T 天 market features
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if version == "v2":
        from code.models.master_v2 import MASTERv2 as MasterCls, T
        ckpt_path = OUTPUT / "checkpoints" / "master_v2.pt"
        H, nhead, dropout = 96, 4, 0.25
        n_intra, n_inter = 3, 2
        F_weight = 0
    else:
        from code.models.master import MASTER as MasterCls, T, N_WEIGHT
        ckpt_path = OUTPUT / "checkpoints" / "master.pt"
        H, nhead, dropout = 64, 4, 0.2
        n_intra, n_inter = 2, 1
        F_weight = N_WEIGHT if any(c in all_features_df.columns for c in WEIGHT_COLS) else 0

    F_market = len([c for c in market_df.columns if c != "trade_date"])

    if version == "v2":
        model = MasterCls(F_stock=len(FACTOR_COLS), F_market=F_market, H=H, T=T,
                          nhead=nhead, dropout=dropout,
                          n_intra_layers=n_intra, n_inter_layers=n_inter).to(device)
    else:
        model = MasterCls(F_stock=len(FACTOR_COLS), F_market=F_market, H=H, T=T,
                          nhead=nhead, dropout=dropout,
                          n_intra_layers=n_intra, n_inter_layers=n_inter,
                          F_weight=F_weight).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    target_date = int(features_df["trade_date"].iloc[0])

    # Stock features
    codes, X, X_w = _gather_history(features_df["ts_code"].values, target_date, all_features_df, T)
    if len(codes) == 0:
        return np.array([]), np.array([])

    # Market features window
    market_df = market_df.sort_values("trade_date").reset_index(drop=True)
    market_cols = [c for c in market_df.columns if c != "trade_date"]
    market_X_all = market_df[market_cols].fillna(0).values.astype(np.float32)
    market_dates = market_df["trade_date"].values

    # 找 <= target_date 的最近 T 天 market window
    mask = market_dates <= target_date
    if mask.sum() < T:
        print(f"WARN: market history insufficient (<{T} days before {target_date})")
        return np.array([]), np.array([])
    m_end = int(np.where(mask)[0][-1])
    market_window = market_X_all[m_end - T + 1 : m_end + 1]

    X_t = torch.from_numpy(X).to(device)
    m_t = torch.from_numpy(market_window).to(device)
    X_w_t = torch.from_numpy(X_w).to(device) if X_w is not None and F_weight > 0 else None
    with torch.no_grad():
        scores = model(X_t, m_t, X_w_t).cpu().numpy()
    return codes, scores


def predict_ensemble(features_df, all_features_df, market_df, weights=None):
    """三模型 rank-ensemble (按当日 rank 标准化后加权)"""
    weights = weights or {"master": 0.5, "lgbm": 0.3, "gru": 0.2}

    codes_l, s_l = predict_lgbm(features_df)
    codes_g, s_g = predict_gru(features_df, all_features_df, T=20)
    codes_m, s_m = predict_master(features_df, all_features_df, market_df, version="v1")

    def to_rank_pct(codes, scores):
        s = pd.Series(scores, index=codes)
        return s.rank(pct=True)

    df = pd.DataFrame({
        "lgbm": to_rank_pct(codes_l, s_l),
        "gru": to_rank_pct(codes_g, s_g),
        "master": to_rank_pct(codes_m, s_m),
    })
    df = df.dropna(how="all")
    # 缺失模型权重重分配: 用 mean 平均填 NaN (各模型 rank 都在 [0,1])
    df = df.fillna(df.mean())

    score = (
        weights["master"] * df["master"]
        + weights["lgbm"] * df["lgbm"]
        + weights["gru"] * df["gru"]
    )
    return score.index.values, score.values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=int, required=True, help="信号日期 YYYYMMDD")
    parser.add_argument("--model", default="master",
                        choices=["lgbm", "gru", "master", "master_v2", "master_v3", "ensemble"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--portfolio", type=str, default="",
                        help="当前持仓 ts_code (逗号分隔)")
    args = parser.parse_args()

    print(f"Generating signal: model={args.model} date={args.date} n={args.n} k={args.k}")

    feats = pd.read_parquet(CACHE / "features.parquet")
    day_feats = feats[feats["trade_date"] == args.date].copy()
    if len(day_feats) == 0:
        print(f"ERROR: no features for {args.date}")
        return
    print(f"  features for date: {len(day_feats)} stocks")

    if args.model == "lgbm":
        codes, scores = predict_lgbm(day_feats)
    elif args.model == "gru":
        codes, scores = predict_gru(day_feats, feats, T=20)
    elif args.model == "master":
        market = pd.read_parquet(CACHE / "market_features.parquet")
        codes, scores = predict_master(day_feats, feats, market, version="v1")
    elif args.model == "master_v2":
        market = pd.read_parquet(CACHE / "market_features.parquet")
        codes, scores = predict_master(day_feats, feats, market, version="v2")
    elif args.model == "master_v3":
        market = pd.read_parquet(CACHE / "market_features.parquet")
        codes, scores = predict_master_multiseed(day_feats, feats, market)
    elif args.model == "ensemble":
        market = pd.read_parquet(CACHE / "market_features.parquet")
        codes, scores = predict_ensemble(day_feats, feats, market)
    else:
        raise ValueError(args.model)

    if len(codes) == 0:
        print("ERROR: no valid predictions")
        return

    score_map = dict(zip(codes, scores))
    day_feats["score"] = day_feats["ts_code"].map(score_map)
    day_feats = day_feats.dropna(subset=["score"])
    day_feats = day_feats.sort_values("score", ascending=False)
    print(f"  predicted: {len(day_feats)} stocks")

    # 过滤器: ST/微盘/低流动性/新股 → 排除出买入候选
    from code.live.signal_filter import filter_signals, load_panel_day
    panel_day = load_panel_day(args.date)
    scored = day_feats.set_index("ts_code")[["score"]]
    passed, excluded, risk_flags = filter_signals(scored, panel_day)
    if len(excluded) > 0:
        print(f"  filter excluded: {len(excluded)} stocks "
              f"(ST={len(excluded[excluded['filter_reason']=='ST'])}, "
              f"illiquid={len(excluded[excluded['filter_reason'].str.startswith('换手')])}, ...)")
    # 用过滤后的得分排序 (已持仓的不受过滤影响)
    basic = pd.read_csv(ROOT / "basic.csv")[["ts_code", "name"]]
    day_feats = day_feats.merge(basic, on="ts_code", how="left")

    day_feats["_pass_filter"] = day_feats["ts_code"].isin(passed.index)
    day_feats_filtered = day_feats[day_feats["_pass_filter"]].copy()

    # 自动加载持仓状态 (如果未显式传 --portfolio)
    portfolio = set(args.portfolio.split(",")) if args.portfolio else set()
    portfolio = {p.strip() for p in portfolio if p.strip()}

    state_file = OUTPUT / "state" / "current_holdings.json"
    if not portfolio and state_file.exists():
        import json
        state = json.load(open(state_file, encoding="utf-8"))
        held = state.get("stocks", {})
        if held:
            portfolio = set(held.keys())
            print(f"  Auto-loaded {len(portfolio)} holdings from {state_file}")

    if not portfolio:
        action_df = day_feats_filtered.head(args.n).copy()
        action_df["action"] = "buy"
        print(f"\nInit position: BUY {len(action_df)} stocks (filtered)")
    else:
        # top-N 基于全部股票排序 (持仓股即使被过滤也应保留)
        top_n_codes = set(day_feats.head(args.n)["ts_code"].values)
        held_in_top = portfolio & top_n_codes
        held_out = portfolio - top_n_codes
        missing_from_signal = portfolio - set(day_feats["ts_code"].values)

        # SELL: 不在 top-N 的持仓, 按得分从低到高, 最多卖 k 只
        zombie_sells = list(missing_from_signal)
        sell_candidates = day_feats[day_feats["ts_code"].isin(
            held_out - missing_from_signal
        )].copy()
        sell_candidates = sell_candidates.sort_values("score", ascending=True)
        k_for_normal = max(0, args.k - len(zombie_sells))
        sell_candidates = sell_candidates.head(k_for_normal)

        sell_rows = []
        if not sell_candidates.empty:
            sell_candidates["action"] = "sell"
            sell_rows.append(sell_candidates[["action", "ts_code", "name", "score"]])
        name_map = dict(zip(basic["ts_code"], basic["name"]))
        for code in zombie_sells:
            sell_rows.append(pd.DataFrame(
                [{"action": "sell", "ts_code": code,
                  "name": name_map.get(code, "?"), "score": float("nan")}]
            ))

        # BUY: 从过滤后候选池选, 显式排序确保取最高分
        buy_slots = min(args.k - len(sell_candidates), args.n - len(held_in_top))
        buy_slots = max(0, buy_slots)
        buy_pool = day_feats_filtered[
            day_feats_filtered["ts_code"].isin(top_n_codes - portfolio)
        ]
        buy_candidates = buy_pool.sort_values("score", ascending=False).head(buy_slots).copy()
        buy_candidates["action"] = "buy"

        parts = []
        if sell_rows:
            parts.append(pd.concat(sell_rows, ignore_index=True)
                         if len(sell_rows) > 1 else sell_rows[0])
        if not buy_candidates.empty:
            parts.append(buy_candidates[["action", "ts_code", "name", "score"]])

        action_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["action", "ts_code", "name", "score"])
        n_sell = sum(1 for _, r in action_df.iterrows() if r["action"] == "sell")
        n_buy = sum(1 for _, r in action_df.iterrows() if r["action"] == "buy")
        n_hold = len(held_in_top)
        print(f"\nRebalance: BUY {n_buy} / SELL {n_sell} / HOLD {n_hold}")

    action_df = action_df[["action", "ts_code", "name", "score"]]
    out_path = OUTPUT / "signals" / f"{args.date}_{args.model}.csv"
    action_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")
    print(action_df.to_string(index=False))


if __name__ == "__main__":
    main()
