import pandas as pd; import numpy as np; import torch; from pathlib import Path
from code.models.master import MASTER, T, N_WEIGHT, FACTOR_COLS, N_FEAT

CACHE = Path('cache'); OUTPUT = Path('output')
WEIGHT_COLS = ['hs300_weight', 'hs300_dweight', 'cyb_weight']
device = 'cuda'

feats = pd.read_parquet(CACHE / 'features.parquet')
market_full = pd.read_parquet(CACHE / 'market_features.parquet')
basic = pd.read_csv('basic.csv')[['ts_code', 'name', 'industry']]
day_feats = feats[feats['trade_date'] == 20260529].copy()
hist = feats[feats['trade_date'] <= 20260529].sort_values(['ts_code', 'trade_date'])

weight_cols = [c for c in WEIGHT_COLS if c in hist.columns]
cols = FACTOR_COLS + weight_cols
codes = day_feats['ts_code'].values
g = hist.groupby('ts_code', sort=False)
X_list, valid_codes = [], []
for code in codes:
    if code in g.indices:
        grp = hist.iloc[g.indices[code]]
        if len(grp) >= T:
            X_list.append(grp.tail(T)[cols].fillna(0).values.astype(np.float32))
            valid_codes.append(code)
X_all = np.stack(X_list)
X = X_all[:, :, :len(FACTOR_COLS)]
X_w = X_all[:, :, len(FACTOR_COLS):]

market = market_full.sort_values('trade_date').reset_index(drop=True)
all_mc = [c for c in market.columns if c != 'trade_date']
old_mc = [c for c in all_mc if c != 'news_count']
market_X_all = market[all_mc].fillna(0).values.astype(np.float32)
old_market_X = market[old_mc].fillna(0).values.astype(np.float32)
m_end = int(np.where(market['trade_date'].values <= 20260529)[0][-1])
mw_new = market_X_all[m_end - T + 1 : m_end + 1]
mw_old = old_market_X[m_end - T + 1 : m_end + 1]

configs = [
    ('v3(今天)', OUTPUT / 'checkpoints' / 'master.pt', True, True),
    ('v2(昨天)', OUTPUT / 'v2_weight' / 'checkpoints' / 'master.pt', True, False),
    ('v1(前天)', OUTPUT / 'v1_20f' / 'checkpoints' / 'master.pt', False, False),
]

all_ranks = {}
for name, ckpt, has_w, use_news in configs:
    Fw = N_WEIGHT if has_w else 0; Fm = 13 if use_news else 12
    mw = mw_new if use_news else mw_old
    model = MASTER(F_stock=N_FEAT, F_market=Fm, H=64, T=T, nhead=4, dropout=0.2,
                   n_intra_layers=2, n_inter_layers=1, F_weight=Fw).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True)); model.eval()
    X_t = torch.from_numpy(X).to(device); m_t = torch.from_numpy(mw).to(device)
    X_w_t = torch.from_numpy(X_w).to(device) if has_w else None
    with torch.no_grad():
        scores = model(X_t, m_t, X_w_t).cpu().numpy()
    df = pd.DataFrame({'ts_code': valid_codes, 'score': scores})
    df = df.merge(basic, on='ts_code', how='left')
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    all_ranks[name] = df

models = list(all_ranks.keys())

# Top-N unique stocks
for N in [10, 20]:
    print(f'=== Top-{N} 独有股 (只在一个模型 top-{N} 中出现) ===')
    for m in models:
        my_set = set(all_ranks[m].head(N)['ts_code'])
        other_sets = set()
        for om in models:
            if om != m:
                other_sets |= set(all_ranks[om].head(N)['ts_code'])
        only_mine = my_set - other_sets
        if only_mine:
            for c in sorted(only_mine):
                row = all_ranks[m][all_ranks[m]['ts_code']==c].iloc[0]
                ranks_other = []
                for om in models:
                    if om != m:
                        r = int(all_ranks[om][all_ranks[om]['ts_code']==c].index[0]) + 1
                        ranks_other.append(f'{om}={r}')
                info = f'  [{m}] #{int(row.name)} {row["ts_code"]} {row["name"]:<8s} {row["industry"]:<8s}  ({"; ".join(ranks_other)})'
                print(info)
    print()

# Biggest rank divergence
print('=== 排名分歧最大 (top-50 内, 极差>=15) ===')
v3, v2, v1 = all_ranks['v3(今天)'], all_ranks['v2(昨天)'], all_ranks['v1(前天)']
top50_codes = set(v3.head(50)['ts_code']) | set(v2.head(50)['ts_code']) | set(v1.head(50)['ts_code'])
divergence = []
for c in top50_codes:
    ranks = []
    for df in [v3, v2, v1]:
        match = df[df['ts_code']==c]
        ranks.append(int(match.index[0]) + 1 if len(match) > 0 else 999)
    diff = max(ranks) - min(ranks)
    if diff >= 15:
        divergence.append((c, ranks, diff))

divergence.sort(key=lambda x: x[2], reverse=True)
fmt = '{:<12} {:<10} {:>5} {:>5} {:>5} {:>5}  {}'
print(fmt.format('代码','名称','v3','v2','v1','极差','行业'))
print('-'*55)
for c, ranks, diff in divergence[:20]:
    name = v3[v3['ts_code']==c]['name'].values[0] if len(v3[v3['ts_code']==c])>0 else '?'
    ind = v3[v3['ts_code']==c]['industry'].values[0] if len(v3[v3['ts_code']==c])>0 else '?'
    print(fmt.format(c, name, str(ranks[0]), str(ranks[1]), str(ranks[2]), str(diff), ind))
