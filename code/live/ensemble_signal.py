"""
Top-5种子集成推理: rank-percentile平均

用法:
  python -m code.live.ensemble_signal --date 20260529 --n 10
"""
import argparse, numpy as np, pandas as pd, torch
from pathlib import Path

from code.config import ROOT, CACHE, OUTPUT, FACTOR_COLS, WEIGHT_COLS, T, N_WEIGHT

(OUTPUT / "signals").mkdir(parents=True, exist_ok=True)
from code.models.master import MASTER

# Top-5 distinct seeds from v3 search (by Sharpe, skipping HP variants of same seed)
ENSEMBLE_SEEDS = [
    ("seed128_H64_alpha0.6_dropout0.2_lr0.0003", "seed128", 2.57),
    ("seed1024", "seed1024", 2.39),
    ("seed99", "seed99", 2.32),
    ("seed8", "seed8", 1.95),
    ("seed4096", "seed4096", 1.85),
]
CKPT_DIR = OUTPUT / "v3_multi_train" / "checkpoints"


def score_all(date, device="cuda"):
    feats = pd.read_parquet(CACHE / "features.parquet")
    market = pd.read_parquet(CACHE / "market_features.parquet")
    basic = pd.read_csv(ROOT / "basic.csv")[['ts_code','name']]

    day_feats = feats[feats['trade_date']==date].copy()
    hist = feats[feats['trade_date']<=date].sort_values(['ts_code','trade_date'])
    cols = FACTOR_COLS + [c for c in WEIGHT_COLS if c in hist.columns]
    codes = day_feats['ts_code'].values
    g = hist.groupby('ts_code', sort=False)
    X_list, valid_codes = [], []
    for code in codes:
        if code in g.indices:
            grp = hist.iloc[g.indices[code]]
            if len(grp) >= T: X_list.append(grp.tail(T)[cols].fillna(0).values.astype(np.float32)); valid_codes.append(code)
    X_all = np.stack(X_list); X = X_all[:,:,:len(FACTOR_COLS)]
    X_w = X_all[:,:,len(FACTOR_COLS):]

    market = market.sort_values('trade_date').reset_index(drop=True)
    market_cols = [c for c in market.columns if c != 'trade_date']
    m_X = market[market_cols].fillna(0).values.astype(np.float32)
    m_end = int(np.where(market['trade_date'].values<=date)[0][-1])
    mw = m_X[m_end-T+1:m_end+1]

    all_scores = {}
    for tag, seed, sharpe_val in ENSEMBLE_SEEDS:
        ckpt_path = CKPT_DIR / f"master_{tag}.pt"
        if not ckpt_path.exists():
            print(f"  WARN: {ckpt_path} missing, skip {tag}")
            continue

        model = MASTER(F_stock=len(FACTOR_COLS), F_market=len(market_cols), H=64, T=T,
                       nhead=4, dropout=0.2, n_intra_layers=2, n_inter_layers=1,
                       F_weight=N_WEIGHT).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()

        X_t = torch.from_numpy(X).to(device)
        m_t = torch.from_numpy(mw).to(device)
        X_w_t = torch.from_numpy(X_w).to(device)
        with torch.no_grad():
            scores = model(X_t, m_t, X_w_t).cpu().numpy()

        s = pd.Series(scores, index=valid_codes)
        s_rank = s.rank(pct=True)  # normalize to [0,1]
        all_scores[tag] = s_rank
        print(f"  {tag:50s} (Sharpe={sharpe_val}) scored {len(valid_codes)} stocks")

    if not all_scores:
        print("ERROR: no models loaded"); return

    # Rank-percentile average
    df = pd.DataFrame(all_scores)
    df['ensemble_score'] = df.mean(axis=1)
    df = df.sort_values('ensemble_score', ascending=False)
    df = df.merge(basic, left_index=True, right_on='ts_code', how='left')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=int, required=True)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ensemble: {len(ENSEMBLE_SEEDS)} models, date={args.date}, device={device}")

    df = score_all(args.date, device)
    if df is None: return

    top = df.head(args.n)
    print(f"\nTop-{args.n}:")
    for i, (_, r) in enumerate(top.iterrows()):
        print(f"  {i+1}. {r['ts_code']} {r['name']:<8s}  {r['ensemble_score']:.4f}")

    # Save
    out = top[['ts_code','name','ensemble_score']].copy()
    out.columns = ['ts_code','name','score']
    out['action'] = 'buy'
    out_path = OUTPUT / "signals" / f"{args.date}_ensemble.csv"
    out[['action','ts_code','name','score']].to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
