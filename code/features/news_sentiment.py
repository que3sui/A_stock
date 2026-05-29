"""
市场级新闻特征: 从 news/ 目录计算每日全市场新闻统计。

输出 (RAW值, 不做标准化 — 由 market_features.py 统一做 train-only Z-score):
  - news_count: 每日新闻总数 (市场活跃度)
  - news_sentiment_mean: 每日情感均值
  - news_sentiment_std: 每日情感离散度

情感打分: 中文金融情感词典法 (正/负词计数), 避免 FinBERT 预训练前瞻偏差。

Usage:
  from code.features.news_sentiment import compute_market_news_features
  df = compute_market_news_features(data_root)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 中文金融情感词典 (简化版, 可扩展)
POS_WORDS = {
    "涨", "增", "利", "赢", "升", "突破", "强势", "反弹", "新高",
    "向好", "改善", "回暖", "超预期", "利好", "盈利", "增长", "上升",
    "上涨", "大涨", "暴涨", "拉升", "走高", "走强", "领涨", "涨停",
    "净流入", "加仓", "买入", "增持", "回购", "分红", "派息",
    "扩张", "签约", "中标", "获批", "量产", "交付",
}
NEG_WORDS = {
    "跌", "降", "亏", "损", "下滑", "下跌", "大跌", "暴跌", "走低",
    "走弱", "领跌", "跌停", "风险", "危机", "暴雷", "违约", "亏损",
    "净流出", "减仓", "卖出", "减持", "套现", "质押", "预警",
    "萎缩", "裁员", "停产", "退市", "处罚", "调查", "诉讼",
    "弱势", "低迷", "承压", "恶化", "不及预期", "利空",
}


def article_sentiment(title, content):
    """对单条新闻做情感打分 [-1, 1]"""
    text = f"{title} {content}" if isinstance(content, str) else str(title)
    pos = sum(1 for w in POS_WORDS if w in text)
    neg = sum(1 for w in NEG_WORDS if w in text)
    return (pos - neg) / (pos + neg + 1)


def compute_market_news_features(data_root):
    """
    读取所有 news CSV, 计算每日市场级特征。

    Args:
        data_root: 项目根目录 (含 news/ 子目录)

    Returns:
        DataFrame [trade_date, news_count, news_sentiment_mean, news_sentiment_std]
        trade_date 为 int (YYYYMMDD), 特征为 float32
    """
    news_dir = Path(data_root) / "news"
    files = sorted(news_dir.glob("*.csv"))
    if not files:
        print("  WARN: no news files found")
        return pd.DataFrame(columns=["trade_date", "news_count",
                                     "news_sentiment_mean", "news_sentiment_std"])

    rows = []
    for f in tqdm(files, desc="news", unit="file"):
        try:
            date = int(f.stem)
        except ValueError:
            continue
        df = pd.read_csv(f)
        sentiments = []
        for _, row in df.iterrows():
            sentiments.append(article_sentiment(row.get("title", ""),
                                                row.get("content", "")))
        sentiments = np.array(sentiments, dtype=np.float32)
        rows.append({
            "trade_date": date,
            "news_count": len(sentiments),
            "news_sentiment_mean": float(np.mean(sentiments)),
            "news_sentiment_std": float(np.std(sentiments)) if len(sentiments) > 1 else 0.0,
        })

    out = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    for c in ["news_count", "news_sentiment_mean", "news_sentiment_std"]:
        out[c] = out[c].astype("float32")
    print(f"  news features: {len(out)} days, "
          f"avg {out['news_count'].mean():.0f} articles/day")
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    ROOT = Path(__file__).resolve().parents[2]
    df = compute_market_news_features(ROOT)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(df.describe())
