"""
信号后处理过滤器: 在模型得分基础上剔除不满足交易条件的股票

Usage:
  from code.live.signal_filter import filter_signals, load_panel_day

  panel_day = load_panel_day(20260605)
  passed, excluded, flags = filter_signals(scores_df, panel_day)

三层过滤:
  1. 自动硬过滤 — 基于 panel 数据的规则 (ST/市值/换手率)
  2. 风险标记 — 提示但不剔除
  3. 用户黑名单 — 手动维护
"""
import pandas as pd
from code.config import CACHE

# === 可调阈值 (基于2026年量化行业最佳实践) ===
MIN_LIST_DAYS = 60         # 最少上市天数
MIN_DAILY_AMOUNT = 3000e4  # 日成交额底线 3000万元

# 市值分层换手率: (下限≤市值<上限, 最低换手率%)
# 区间左闭右开: lo <= circ_mv < hi
TURNOVER_TIERS = [
    (0,         500_000,  1.0),   # <50亿: 换手≥1%
    (500_000,   2_000_000, 0.7),  # 50-200亿: 换手≥0.7%
    (2_000_000, 10_000_000, 0.5), # 200-1000亿: 换手≥0.5%
    (10_000_000, float("inf"), 0.3), # ≥1000亿大盘蓝筹: 换手≥0.3%即可
]
MAX_TURNOVER = 20.0          # 换手>20%: 炒作/出货信号 (不受豁免影响)
MIN_DAILY_VOLUME = 5e8       # 日成交额>5亿: 豁免最低换手率检查 (MAX_TURNOVER仍生效)
FLAG_LOW_TURNOVER = 1.0    # 换手率低于此值标记警告
FLAG_EXTREME_MOVE = 9.0    # 涨跌幅绝对值超过此值标记

USER_BLACKLIST: frozenset[str] = frozenset()


def _get_turnover_threshold(circ_mv):
    """根据流通市值(万元)返回换手率阈值"""
    for lo, hi, threshold in TURNOVER_TIERS:
        if lo <= circ_mv < hi:
            return threshold
    return 1.0


def load_panel_day(date):
    """加载当日 panel 数据用于过滤"""
    cols = ["trade_date", "ts_code", "circ_mv", "turnover_rate", "vol",
            "is_st", "list_days", "close", "pct_chg"]
    panel = pd.read_parquet(CACHE / "panel.parquet", columns=cols)
    return panel[panel["trade_date"] == date].set_index("ts_code")


def filter_signals(scores_df, panel_day, blacklist=None):
    """
    完整的信号过滤流水线

    scores_df: DataFrame, index=ts_code, 至少包含 'score' 列
    panel_day: DataFrame, index=ts_code (来自 load_panel_day)
    blacklist: set of ts_code to exclude

    Returns: (passed_df, excluded_df, flags_df)
      passed_df: 通过过滤的股票 (含 panel 数据列)
      excluded_df: 被剔除的股票 (含 filter_reason 列)
      flags_df: 风险标记 (含 low_turnover, extreme_move 布尔列)
    """
    if blacklist is None:
        blacklist = set(USER_BLACKLIST)  # defensive copy

    df = scores_df.join(panel_day, how="left")

    # NaN score → exclude (model error / missing data)
    if df["score"].isna().any():
        df = df[df["score"].notna()].copy()

    # --- 第一层: 硬过滤 ---
    reasons: dict[str, str] = {}
    rules = [
        (df["is_st"].fillna(True) == True, "ST"),
        (df["list_days"].fillna(0) < MIN_LIST_DAYS, f"上市<{MIN_LIST_DAYS}天"),
    ]
    for mask, label in rules:
        reasons.update(dict.fromkeys(df.loc[mask].index, label))

    # 成交额底线: vol(手) * close * 100 = 元
    daily_value = df["vol"].fillna(0) * df["close"].fillna(0) * 100
    low_amount = daily_value < MIN_DAILY_AMOUNT
    reasons.update(dict.fromkeys(df.loc[low_amount].index, f"日成交<{MIN_DAILY_AMOUNT/1e4:.0f}万"))

    # 市值分层换手率 (日成交>5亿豁免最低换手检查, MAX_TURNOVER 不受豁免)
    for idx, row in df.iterrows():
        if idx in reasons:
            continue
        if daily_value[idx] >= MIN_DAILY_VOLUME:
            continue  # 日成交充足, 换手率再低也能流畅进出
        mv = row.get("circ_mv", 0) or 0
        to = row.get("turnover_rate", 0) or 0
        threshold = _get_turnover_threshold(mv)
        if to < threshold:
            reasons[idx] = f"换手<{threshold}%(市值{mv/1e4:.0f}亿)"

    # 换手上限 (炒作/出货)
    extreme_to = df["turnover_rate"].fillna(0) > MAX_TURNOVER
    for idx in df.loc[extreme_to].index:
        if idx not in reasons:
            reasons[idx] = f"换手>{MAX_TURNOVER}%(异常)"

    for c in blacklist:
        if c in df.index:
            reasons.setdefault(c, "用户黑名单")

    excluded = df.loc[list(reasons.keys())].copy()
    excluded["filter_reason"] = excluded.index.map(reasons)
    passed = df.drop(excluded.index)  # errors="raise" catches index mismatches

    # --- 第二层: 风险标记 ---
    flags = pd.DataFrame(index=passed.index)
    flags["low_turnover"] = passed["turnover_rate"].lt(FLAG_LOW_TURNOVER)
    flags["extreme_move"] = passed["pct_chg"].abs().gt(FLAG_EXTREME_MOVE)

    return passed, excluded, flags
