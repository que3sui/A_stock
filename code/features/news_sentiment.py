"""
市场级新闻特征: 从 news/ 目录计算每日新闻统计。

情感打分: FinBERT 中文金融情感模型 (luhangmei/FinBERT-Chinese) + GPU batch inference.
首次运行自动下载模型 (~400MB), 缓存到 HuggingFace cache.

Output (RAW, 由 market_features.py 做 train-only Z-score):
  - news_count: 每日新闻总数
  - news_sentiment_mean: 每日情感均值 [-1, 1]
  - news_sentiment_std: 每日情感离散度
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# 中文金融 FinBERT (HuggingFace), 首次运行自动下载
MODEL_NAME = "luhangmei/FinBERT-Chinese"
BATCH_SIZE = 128
MAX_LEN = 256  # 新闻标题+内容截断长度

_sentiment_pipeline = None


def _get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        device = 0 if torch.cuda.is_available() else -1
        print(f"  Loading FinBERT model on {'GPU' if device==0 else 'CPU'} ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _sentiment_pipeline = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer,
            device=device, batch_size=BATCH_SIZE, truncation=True, max_length=MAX_LEN,
        )
    return _sentiment_pipeline


def _batch_sentiment(texts):
    """批量情感推理, 返回 [-1, 1] 分数"""
    if not texts:
        return np.array([], dtype=np.float32)

    pipe = _get_pipeline()
    results = pipe(texts)

    scores = np.zeros(len(texts), dtype=np.float32)
    for i, r in enumerate(results):
        s = float(r["score"])
        scores[i] = s if r["label"].upper() in ("POSITIVE", "LABEL_1", "1") else -s
    return scores


def compute_market_news_features(data_root, use_finbert=True):
    """读取所有 news CSV, 计算每日市场级特征。

    Args:
        data_root: 项目根目录
        use_finbert: True=FinBERT情感, False=仅news_count(快速模式)
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
        count = len(df)

        if use_finbert and count > 0:
            # 构建 (title + content) 截断文本
            texts = []
            for _, row in df.iterrows():
                t = str(row.get("title", ""))
                c = str(row.get("content", "")) if pd.notna(row.get("content")) else ""
                text = f"{t} {c}"[:MAX_LEN * 2]  # double max_len for Chinese chars
                texts.append(text)

            scores = _batch_sentiment(texts)
            mean_s = float(np.mean(scores))
            std_s = float(np.std(scores)) if len(scores) > 1 else 0.0
        else:
            mean_s, std_s = 0.0, 0.0

        rows.append({
            "trade_date": date,
            "news_count": count,
            "news_sentiment_mean": mean_s,
            "news_sentiment_std": std_s,
        })

    out = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    for c in ["news_count", "news_sentiment_mean", "news_sentiment_std"]:
        out[c] = out[c].astype("float32")
    print(f"  news features: {len(out)} days, avg {out['news_count'].mean():.0f} articles/day")
    return out


# 快速模式: 仅 news_count (不改架构, 可叠加到 market_features 快速验证)
def compute_news_count(data_root):
    """仅统计每日新闻数 (不跑 FinBERT)"""
    return compute_market_news_features(data_root, use_finbert=False)


if __name__ == "__main__":
    import time
    ROOT = Path(__file__).resolve().parents[2]
    for use_fb in [False, True]:
        t0 = time.time()
        print(f"\n{'='*60}\nuse_finbert={use_fb}")
        df = compute_market_news_features(ROOT, use_finbert=use_fb)
        print(f"Time: {time.time()-t0:.0f}s")
        print(df.describe())
        if use_fb:
            print(f"Sentiment range: [{df.news_sentiment_mean.min():.3f}, {df.news_sentiment_mean.max():.3f}]")
