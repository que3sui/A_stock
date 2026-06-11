"""分析 Intra-stock Transformer 的 attention 权重: 模型学到时序滤波了吗?"""
import pandas as pd; import numpy as np; import torch
from pathlib import Path

from code.config import ROOT, CACHE, OUTPUT, FACTOR_COLS, WEIGHT_COLS, T, N_WEIGHT
from code.models.master import MASTER
device = 'cuda'

# 1. 加载数据
feats = pd.read_parquet(CACHE / 'features.parquet')
market = pd.read_parquet(CACHE / 'market_features.parquet').sort_values('trade_date').reset_index(drop=True)
basic = pd.read_csv('basic.csv')[['ts_code','name']]

day_feats = feats[feats['trade_date']==20260529].copy()
hist = feats[feats['trade_date']<=20260529].sort_values(['ts_code','trade_date'])
cols = FACTOR_COLS + [c for c in WEIGHT_COLS if c in hist.columns]
codes = day_feats['ts_code'].values
g = hist.groupby('ts_code', sort=False)
X_list, valid_codes = [], []
for code in codes:
    if code in g.indices:
        grp = hist.iloc[g.indices[code]]
        if len(grp) >= T: X_list.append(grp.tail(T)[cols].fillna(0).values.astype(np.float32)); valid_codes.append(code)
X_all = np.stack(X_list)
X = X_all[:,:,:len(FACTOR_COLS)]; X_w = X_all[:,:,len(FACTOR_COLS):]

market_cols = [c for c in market.columns if c != 'trade_date']
m_X = market[market_cols].fillna(0).values.astype(np.float32)
m_end = int(np.where(market['trade_date'].values<=20260529)[0][-1])
mw = m_X[m_end-T+1:m_end+1]

# 加载模型
model = MASTER(F_stock=len(FACTOR_COLS), F_market=len(market_cols), H=64, T=T,
               nhead=4, dropout=0.2, n_intra_layers=2, n_inter_layers=1,
               F_weight=N_WEIGHT).to(device)
model.load_state_dict(torch.load(OUTPUT / 'checkpoints' / 'master.pt', weights_only=True))
model.eval()

# 只 hook temp_attn — 这是时序聚合的关键层，直接告诉我们"看哪几天"
temp_attn_weights = []
def hook_temp(module, input, output):
    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
        temp_attn_weights.append(output[1].detach().cpu().numpy())

h_temp = model.temp_attn.register_forward_hook(hook_temp)

X_t = torch.from_numpy(X).to(device); m_t = torch.from_numpy(mw).to(device)
X_w_t = torch.from_numpy(X_w).to(device)

# 2. 代表性个股
sample_codes = ['000100.SZ','600519.SH','300750.SZ','601857.SH','000661.SZ','600157.SH','603160.SH']
sample_names = {r['ts_code']:r['name'] for _,r in basic[basic['ts_code'].isin(sample_codes)].iterrows()}

print('=== 代表性个股 temp_attn (时序聚合注意力) ===')
print(f'{"代码":<12} {"名称":<8} {"峰值天":<8} {"远/近比":<8} {"HHI":<8} {"20天权重分布"}')
print('-'*90)
for code in sample_codes:
    idx = valid_codes.index(code) if code in valid_codes else -1
    if idx < 0: continue
    temp_attn_weights.clear()
    x_one = X_t[idx:idx+1]; xw_one = X_w_t[idx:idx+1]
    with torch.no_grad():
        _ = model(x_one, m_t, xw_one)
    if temp_attn_weights:
        tw = temp_attn_weights[0][0,0,:]  # [T]
        peak = T - 1 - int(np.argmax(tw))
        decay = float(tw[-1] / (tw[0] + 1e-8))
        hhi = float((tw**2).sum() / (tw.sum()**2 + 1e-8))
        bars = ''.join('█' if w > 0.08 else ('▓' if w > 0.06 else ('▒' if w > 0.04 else '░')) for w in tw)
        print(f'{code:<12} {sample_names.get(code,"?"):<8} t-{peak:<7} {decay:<8.2f} {hhi:<8.3f} {bars}')

# 3. 全市场统计
print(f'\n=== 100只随机样本统计 ===')
all_peaks, all_decays, all_hhi = [], [], []
rng = np.random.default_rng(42)
sample_indices = rng.choice(len(valid_codes), min(100, len(valid_codes)), replace=False)

for idx in sample_indices:
    temp_attn_weights.clear()
    x_one = X_t[idx:idx+1]; xw_one = X_w_t[idx:idx+1]
    with torch.no_grad():
        _ = model(x_one, m_t, xw_one)
    if temp_attn_weights:
        tw = temp_attn_weights[0][0,0,:]
        all_peaks.append(T-1-int(np.argmax(tw)))
        all_decays.append(float(tw[-1] / (tw[0]+1e-8)))
        all_hhi.append(float((tw**2).sum() / (tw.sum()**2 + 1e-8)))

if all_peaks:
    peaks_arr = np.array(all_peaks)
    unique, counts = np.unique(peaks_arr, return_counts=True)
    mode_peak = unique[np.argmax(counts)]
    recent = (peaks_arr <= 3).mean() * 100  # % with peak in last 3 days

    print(f'  峰值位置: mean=t-{np.mean(all_peaks):.1f}  median=t-{np.median(all_peaks):.1f}  mode=t-{mode_peak}')
    print(f'  峰值在近3天内的比例: {recent:.0f}%')
    print(f'  decay ratio: mean={np.mean(all_decays):.2f} (>1=近期偏重)')
    print(f'  HHI集中度:  mean={np.mean(all_hhi):.3f}  (0.05=均匀, >0.15=集中)')
    print()
    if np.mean(all_decays) > 1.5:
        print('  >>> 结论: 模型已学会 EMA 模式, 极大偏重近期。不需要额外加 EMA 因子。')
    elif np.mean(all_hhi) < 0.08:
        print('  >>> 结论: 注意力接近均匀分布, 模型没有学到时序滤波。加 EMA/HP filter 因子有价值!')
    else:
        print(f'  >>> 结论: 混合模式。decay={np.mean(all_decays):.2f}, HHI={np.mean(all_hhi):.3f}。')
        print(f'      如需进一步改善, 可加 HP filter 趋势强度因子。')

    # 分布
    print(f'\n  峰值位置分布:')
    for day in range(T):
        cnt = (peaks_arr == day).sum()
        if cnt > 0:
            bar = '█' * cnt
            print(f'    t-{day:2d}: {cnt:3d} {bar}')

h_temp.remove()
