import pandas as pd
import numpy as np

sig = pd.read_csv('C:/Users/停云行玖/Desktop/dp_lab/大作业/A股数据/output/signals/signals_20260601.csv')
sig = sig.sort_values('score', ascending=False).reset_index(drop=True)
sig['rank_idx'] = range(1, len(sig)+1)

df601 = pd.read_csv('C:/Users/停云行玖/Desktop/dp_lab/大作业/A股数据/daily/20260601.csv')
df529 = pd.read_csv('C:/Users/停云行玖/Desktop/dp_lab/大作业/A股数据/daily/20260529.csv')

def get_price(df, code):
    rows = df[df['ts_code'] == code]
    if len(rows):
        return rows.iloc[0]['close']
    return None

# ============================================
# PLAN A
# ============================================
print('='*80)
print('PLAN A -- 模拟账户')
print('='*80)

planA_holdings = {
    '000100.SZ': {'name':'TCL科技', 'lots':23200, 'cost529':4.28},
    '600642.SH': {'name':'申能股份', 'lots':9600, 'cost529':10.32},
    '603160.SH': {'name':'汇顶科技', 'lots':1600, 'cost529':61.01},
    '000661.SZ': {'name':'长春高新', 'lots':1300, 'cost529':73.57},
    '688516.SH': {'name':'奥特维', 'lots':1500, 'cost529':64.49},
    '002179.SZ': {'name':'中航光电', 'lots':2700, 'cost529':36.42},
    '600157.SH': {'name':'永泰能源', 'lots':54700, 'cost529':1.82},
    '601857.SH': {'name':'中国石油', 'lots':9100, 'cost529':10.87},
    '002402.SZ': {'name':'和而泰', 'lots':3600, 'cost529':27.19},
    '600098.SH': {'name':'广州发展', 'lots':12400, 'cost529':8.00},
}

print('\n[5/29持仓 -> 6/1市值]')
total_value = 0
for code, h in planA_holdings.items():
    p601 = get_price(df601, code)
    val = h['lots'] * p601
    total_value += val
    srow = sig[sig['ts_code']==code]
    srank = str(int(srow.iloc[0]['rank_idx'])) if len(srow) else '--'
    print(f'  {code} {h["name"]:6s} {h["lots"]:6d}股  '
          f'5/29=Y{h["cost529"]:.2f}  6/1=Y{p601:.2f}  市值=Y{val:>10,.2f}  rank={srank:>4s}')

print(f'\n持仓总市值: Y{total_value:,.2f}')

# Determine sells
untracked = [c for c in planA_holdings if len(sig[sig['ts_code']==c])==0]
tracked = [c for c in planA_holdings if len(sig[sig['ts_code']==c])>0]
tracked_scores_df = sig[sig['ts_code'].isin(tracked)].sort_values('score')
sells_tracked = list(tracked_scores_df.head(2)['ts_code'])
keeps = [c for c in tracked if c not in sells_tracked]

print('\n--- 卖出 (4只) ---')
sell_cash = 0
for code in untracked:
    h = planA_holdings[code]
    p = get_price(df601, code)
    val = h['lots'] * p
    sell_cash += val
    print(f'  SELL {code} {h["name"]} {h["lots"]}股 x Y{p:.2f} = Y{val:,.2f}  [无模型信号,清仓]')

for code in sells_tracked:
    h = planA_holdings[code]
    p = get_price(df601, code)
    val = h['lots'] * p
    sell_cash += val
    s = sig[sig['ts_code']==code].iloc[0]
    print(f'  SELL {code} {h["name"]} {h["lots"]}股 x Y{p:.2f} = Y{val:,.2f}  [rank={int(s.rank_idx)} score={s.score:.4f}]')

existing_cash = 996582.23 - sum(h['lots']*h['cost529'] for h in planA_holdings.values())
buy_budget = existing_cash + sell_cash
print(f'\n可用资金: 留存Y{existing_cash:,.2f} + 卖出Y{sell_cash:,.2f} = Y{buy_budget:,.2f}')

print('\n--- 保留 (6只) ---')
keep_value = 0
for code in keeps:
    h = planA_holdings[code]
    p = get_price(df601, code)
    val = h['lots'] * p
    keep_value += val
    s = sig[sig['ts_code']==code].iloc[0]
    print(f'  KEEP {code} {h["name"]} {h["lots"]}股 x Y{p:.2f} = Y{val:,.2f}  [rank={int(s.rank_idx)} score={s.score:.4f}]')
print(f'保留总市值: Y{keep_value:,.2f}')

# === BUY 4: smarter allocation ===
not_held = sig[~sig['ts_code'].isin(planA_holdings.keys())]
buy_targets = not_held.head(4)

print(f'\n--- 买入 (4只, n=10回补) ---')
print(f'预算: Y{buy_budget:,.2f}')

# Phase 1: minimum lots
buy_plan = []
for _, r in buy_targets.iterrows():
    price = get_price(df601, r['ts_code'])
    buy_plan.append({
        'code': r['ts_code'], 'name': r['name'], 'industry': r['industry'],
        'price': price, 'lots': 100, 'cost': 100 * price,
        'rank': int(r['rank_idx']), 'score': r['score']
    })

# Phase 2: allocate remaining to cheaper stocks (indices 2,3 = 神华, 宁德时代)
remaining = buy_budget - sum(b['cost'] for b in buy_plan)
# Give ~70% to 神华, ~30% to 宁德时代 (adjusted for lot constraints)
for idx, alloc_pct in [(2, 0.72), (3, 0.28)]:
    b = buy_plan[idx]
    extra_lots = int(remaining * alloc_pct / (b['price'] * 100)) * 100
    b['lots'] += extra_lots
    b['cost'] = b['lots'] * b['price']
    remaining -= extra_lots * b['price']

total_buy_cost = 0
print(f'{"Code":12s} {"Name":8s} {"行业":6s} {"Rank":>5s} {"Price":>8s} {"Lots":>6s} {"Cost":>12s}')
for b in buy_plan:
    print(f'{b["code"]:12s} {b["name"]:8s} {b["industry"]:6s} {b["rank"]:5d} {b["price"]:8.2f} {b["lots"]:6d} {b["cost"]:12,.2f}')
    total_buy_cost += b['cost']

print(f'\n买入合计: Y{total_buy_cost:,.2f}')
print(f'剩余现金: Y{buy_budget - total_buy_cost:,.2f}')
print(f'调仓后总持仓: 6保留(Y{keep_value:,.2f}) + 4新买(Y{total_buy_cost:,.2f}) = Y{keep_value+total_buy_cost:,.2f}')


# ============================================
# PLAN C
# ============================================
print('\n' + '='*80)
print('PLAN C -- 实盘账户')
print('  实盘A: Y61,000 (98.4%)  +  实盘B: Y1,015 (1.6%)')
print('='*80)

planC_holdings = {
    '000100.SZ': {'name':'TCL科技', 'lots':7200, 'cost529':4.28},
    '600157.SH': {'name':'永泰能源', 'lots':17000, 'cost529':1.82},
}

print('\n[5/29持仓 -> 6/1市值]')
for code, h in planC_holdings.items():
    p601 = get_price(df601, code)
    val = h['lots'] * p601
    srow = sig[sig['ts_code']==code]
    srank = str(int(srow.iloc[0]['rank_idx'])) if len(srow) else '--'
    print(f'  {code} {h["name"]} {h["lots"]}股 x Y{p601:.2f} = Y{val:,.2f}  rank={srank}')

# Sell TCL科技, Keep 永泰能源
sell_code = '000100.SZ'
sell_info = planC_holdings[sell_code]
sell_price = get_price(df601, sell_code)
sell_val = sell_info['lots'] * sell_price

keep_code = '600157.SH'
keep_info = planC_holdings[keep_code]
keep_price = get_price(df601, keep_code)
keep_val = keep_info['lots'] * keep_price

existing_c_cash = 62014.99 - sum(h['lots']*h['cost529'] for h in planC_holdings.values())
buy_c_budget = existing_c_cash + sell_val

print(f'\n--- 卖出 ---')
print(f'  SELL {sell_code} {sell_info["name"]} {sell_info["lots"]}股 x Y{sell_price:.2f} = Y{sell_val:,.2f}')
print(f'  (rank=669, 信号弱于阈值)')

print(f'\n--- 保留 ---')
print(f'  KEEP {keep_code} {keep_info["name"]} {keep_info["lots"]}股 x Y{keep_price:.2f} = Y{keep_val:,.2f}')
print(f'  (rank=204, 信号正常)')

# Buy candidates
not_held_c = sig[~sig['ts_code'].isin(['000100.SZ','600157.SH'])]
affordable_codes = []
for c in not_held_c['ts_code'].head(50):
    p = get_price(df601, c)
    if p and p * 100 <= buy_c_budget:
        affordable_codes.append(c)

affordable = not_held_c[not_held_c['ts_code'].isin(affordable_codes)]

print(f'\n--- 买入候选 (预算Y{buy_c_budget:,.2f}) ---')
for _, r in affordable.head(8).iterrows():
    price = get_price(df601, r['ts_code'])
    max_lots = int(buy_c_budget / (price * 100)) * 100
    print(f'  {r["ts_code"]} {r["name"]:8s} {r["industry"]:8s}  rank={int(r["rank_idx"]):4d}  Y{price:>8.2f}  max{max_lots}股')

# == Combined plan (total pool) ==
buy_c = affordable.iloc[0]  # 601857 中国石油
buy_c_code = buy_c['ts_code']
buy_c_name = buy_c['name']
buy_c_price = get_price(df601, buy_c_code)
buy_c_lots = int(buy_c_budget / (buy_c_price * 100)) * 100
buy_c_cost = buy_c_lots * buy_c_price

print(f'\n推荐买入(总池): {buy_c_code} {buy_c_name}  {buy_c_lots}股 x Y{buy_c_price:.2f} = Y{buy_c_cost:,.2f}')
print(f'总池剩余: Y{buy_c_budget - buy_c_cost:,.2f}')

# Split
ratio_a = 61000 / 62014.99
ratio_b = 1014.99 / 62014.99

# Account A
a_lots_buy = int(buy_c_lots * ratio_a / 100) * 100
a_cost_buy = a_lots_buy * buy_c_price
a_lots_keep = int(keep_info['lots'] * ratio_a / 100) * 100
a_val_keep = a_lots_keep * keep_price
# Account A sell: its share of TCL科技
a_lots_sell = int(sell_info['lots'] * ratio_a / 100) * 100
a_val_sell = a_lots_sell * sell_price

# Account B
b_lots_buy = buy_c_lots - a_lots_buy
b_cost_buy = b_lots_buy * buy_c_price
b_lots_keep = keep_info['lots'] - a_lots_keep
b_val_keep = b_lots_keep * keep_price
b_lots_sell = sell_info['lots'] - a_lots_sell
b_val_sell = b_lots_sell * sell_price

# Check if Account B can afford its buy
b_remaining = 1014.99 * (existing_c_cash / 62014.99) + b_val_sell
b_need = b_cost_buy
b_ok = b_remaining >= b_need

print(f'\n[分账户执行]')
print(f'')
print(f'  实盘A (Y61,000, {ratio_a*100:.1f}%):')
print(f'    卖出: {sell_code} {sell_info["name"]} {a_lots_sell}股 x Y{sell_price:.2f} = Y{a_val_sell:,.2f}')
print(f'    保留: {keep_code} {keep_info["name"]} {a_lots_keep}股 x Y{keep_price:.2f} = Y{a_val_keep:,.2f}')
print(f'    买入: {buy_c_code} {buy_c_name} {a_lots_buy}股 x Y{buy_c_price:.2f} = Y{a_cost_buy:,.2f}')
print(f'')
print(f'  实盘B (Y1,015, {ratio_b*100:.1f}%):')
print(f'    卖出: {sell_code} {sell_info["name"]} {b_lots_sell}股 x Y{sell_price:.2f} = Y{b_val_sell:,.2f}')
print(f'    保留: {keep_code} {keep_info["name"]} {b_lots_keep}股 x Y{keep_price:.2f} = Y{b_val_keep:,.2f}')
if b_ok:
    print(f'    买入: {buy_c_code} {buy_c_name} {b_lots_buy}股 x Y{buy_c_price:.2f} = Y{b_cost_buy:,.2f}')
else:
    # Fallback: buy cheapest top-10 stock
    for _, r in affordable.head(10).iterrows():
        p = get_price(df601, r['ts_code'])
        if p * 100 <= b_remaining:
            fallback_code = r['ts_code']
            fallback_name = r['name']
            fallback_price = p
            fallback_lots = int(b_remaining / (p * 100)) * 100
            fallback_cost = fallback_lots * p
            print(f'    买入: {fallback_code} {fallback_name} {fallback_lots}股 x Y{fallback_price:.2f} = Y{fallback_cost:,.2f}')
            print(f'    (601857资金不足Y{b_need-b_remaining:,.0f}, 改用更低价标的)')
            break
