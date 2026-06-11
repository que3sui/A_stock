"""
共享评估指标: IC / RankIC 日度横截面聚合

从 master.py / gru_att.py / mlp_baseline.py / master_v2.py 的 evaluate() 中提取,
消除 4 处完全重复的 IC 聚合逻辑。
"""
import numpy as np


def daily_ic(df):
    """从 (trade_date, score, label) DataFrame 计算日度 IC + RankIC 序列"""
    ics, rank_ics = [], []
    for _, day in df.groupby("trade_date"):
        if len(day) < 30 or day["score"].std() == 0:
            continue
        ics.append(day["score"].corr(day["label"]))
        rank_ics.append(day["score"].rank().corr(day["label"].rank()))
    return ics, rank_ics


def ic_summary(df):
    """计算 IC mean / RankIC mean / RankIC std, 返回三元组"""
    ics, rank_ics = daily_ic(df)
    return float(np.mean(ics)), float(np.mean(rank_ics)), float(np.std(rank_ics))
