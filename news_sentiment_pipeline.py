"""
=============================================================================
 NEWS SENTIMENT PIPELINE — FinBERT Daily Sentiment for Silver Forecasting
=============================================================================
 Model   : ProsusAI/finbert  (Financial BERT, 3-class: positive/negative/neutral)
 Sources : Google News RSS, Yahoo Finance RSS, yfinance ticker news
 Topics  : Silver market | Precious Metals | Federal Reserve
 Output  : Daily-indexed DataFrame (sentiment_score: -1.0 to +1.0)
           Merge-ready with silver_ML_dataset.csv
=============================================================================
"""

import sys
import warnings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import re
import time
import hashlib
from datetime import datetime, timedelta, date, timezone
from typing import Optional

import numpy as np
import pandas as pd
import feedparser
import yfinance as yf


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
FINBERT_MODEL   = "ProsusAI/finbert"
MAX_TOKEN_LEN   = 512          # FinBERT hard limit
BATCH_SIZE      = 16           # Headlines per inference batch
SCORE_MAP       = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

# Lookback window for RSS / yfinance news (days)
NEWS_LOOKBACK_DAYS = 90

# Google News RSS query strings (URL-encoded)
RSS_QUERIES = [
    "Silver+market+price",
    "Precious+Metals+gold+silver",
    "Federal+Reserve+interest+rates+metals",
    "silver+futures+CME",
    "XAG+silver+commodities",
]

# yfinance tickers whose built-in news API we will also scrape
YFINANCE_TICKERS = ["SI=F", "GLD", "SLV", "GC=F"]

# ---------------------------------------------------------------------------
# MODULE 1 — MODEL LOADER (MOCKED)
# ---------------------------------------------------------------------------

_tokenizer = None
_model     = None
_device    = None


def load_finbert(model_name: str = FINBERT_MODEL):
    """Mock loading function to preserve interface compatibility."""
    print("[Lightweight Sentiment] load_finbert called. Skipping loading FinBERT (zero-RAM mode).")
    return None, None, None


# ---------------------------------------------------------------------------
# MODULE 2 — NEWS FETCHERS
# ---------------------------------------------------------------------------

def _parse_date(entry) -> Optional[date]:
    """Extract a date object from a feedparser entry, or return today."""
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            try:
                return date(*val[:3])
            except Exception:
                pass
    # Fall back: try parsing 'published' string
    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if raw:
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw[:len(fmt)], fmt).date()
            except Exception:
                pass
    return date.today()


def fetch_rss_news(lookback_days: int = NEWS_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetches headlines from Google News RSS for each query topic.

    Returns
    -------
    pd.DataFrame with columns: [date, headline, source]
    """
    cutoff  = date.today() - timedelta(days=lookback_days)
    records = []

    for query in RSS_QUERIES:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        try:
            import requests
            resp = requests.get(url, timeout=5)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries:
                title = getattr(entry, "title", "").strip()
                if not title:
                    continue
                pub_date = _parse_date(entry)
                if pub_date < cutoff:
                    continue
                records.append({
                    "date"    : pub_date,
                    "headline": title,
                    "source"  : "google_rss",
                })
        except Exception as e:
            print(f"  [RSS] Warning: could not fetch '{query}': {e}")

    df = pd.DataFrame(records)
    print(f"  [RSS] Fetched {len(df)} headlines from Google News RSS.")
    return df


def fetch_yfinance_news(lookback_days: int = NEWS_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetches news via yfinance .news property for relevant tickers.

    Returns
    -------
    pd.DataFrame with columns: [date, headline, source]
    """
    cutoff  = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    records = []

    for ticker_sym in YFINANCE_TICKERS:
        try:
            ticker = yf.Ticker(ticker_sym)
            news_items = ticker.news or []
            for item in news_items:
                title = item.get("content", {}).get("title") or item.get("title", "")
                title = title.strip()
                if not title:
                    continue
                # Timestamp can be in several formats
                ts = (
                    item.get("content", {}).get("pubDate")
                    or item.get("providerPublishTime")
                )
                try:
                    if isinstance(ts, (int, float)):
                        pub_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()
                    elif isinstance(ts, str):
                        pub_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                    else:
                        pub_date = date.today()
                except Exception:
                    pub_date = date.today()

                if datetime.combine(pub_date, datetime.min.time()).replace(tzinfo=timezone.utc) < cutoff:
                    continue
                records.append({
                    "date"    : pub_date,
                    "headline": title,
                    "source"  : f"yfinance_{ticker_sym}",
                })
        except Exception as e:
            print(f"  [yfinance] Warning: {ticker_sym} news failed: {e}")

    df = pd.DataFrame(records)
    print(f"  [yfinance] Fetched {len(df)} headlines from yfinance tickers.")
    return df


def fetch_all_news(lookback_days: int = NEWS_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Combines all news sources, deduplicates by headline hash, and returns
    a clean DataFrame sorted chronologically.

    Returns
    -------
    pd.DataFrame  columns: [date, headline, source]
                  index  : RangeIndex
    """
    print(f"\n{'='*62}")
    print(f"  NEWS COLLECTION  (lookback: {lookback_days} days)")
    print(f"{'='*62}")

    rss_df = fetch_rss_news(lookback_days)
    yf_df  = fetch_yfinance_news(lookback_days)

    combined = pd.concat([rss_df, yf_df], ignore_index=True)

    if combined.empty:
        print("  [WARN] No news fetched from any source.")
        return combined

    # Deduplicate by MD5 hash of normalised headline text
    combined["_hash"] = combined["headline"].str.lower().str.strip().apply(
        lambda t: hashlib.md5(t.encode()).hexdigest()
    )
    combined.drop_duplicates(subset="_hash", inplace=True)
    combined.drop(columns="_hash", inplace=True)

    combined["date"] = pd.to_datetime(combined["date"])
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"\n  [Total] {len(combined)} unique headlines across all sources.")
    print(f"  [Range] {combined['date'].min().date()} -> {combined['date'].max().date()}")
    return combined


# ---------------------------------------------------------------------------
# MODULE 3 — LIGHTWEIGHT SENTIMENT INFERENCE (ZERO RAM)
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    'up', 'gain', 'rise', 'rising', 'gained', 'gaining', 'gains', 'rises',
    'bullish', 'rally', 'rallied', 'growth', 'grow', 'growing', 'high', 'higher',
    'highest', 'increase', 'increased', 'increasing', 'positive', 'positively',
    'surge', 'surged', 'soar', 'soared', 'soaring', 'strengthen', 'strengthened',
    'strengthening', 'beat', 'beaten', 'beats', 'support', 'supported', 'supporting',
    'profit', 'profitable', 'profits', 'advance', 'advanced', 'advancing', 'peak',
    'outperform', 'outperformed', 'boost', 'boosted', 'boosting', 'optimism', 'optimistic',
    'recovery', 'recover', 'recovered', 'rebounding', 'rebound', 'jump', 'jumped',
    'demand', 'strong', 'stronger', 'bright', 'stimulus'
}

NEGATIVE_WORDS = {
    'down', 'drop', 'dropped', 'dropping', 'drops', 'fall', 'fell', 'falling',
    'falls', 'bearish', 'loss', 'losses', 'lost', 'decline', 'declined', 'declining',
    'declines', 'low', 'lower', 'lowest', 'decrease', 'decreased', 'decreasing',
    'negative', 'negatively', 'slump', 'slumped', 'slumping', 'plummet', 'plummeted',
    'plummeting', 'weaken', 'weakened', 'weakening', 'miss', 'missed', 'missing',
    'resistance', 'deficit', 'retreat', 'retreated', 'plunge', 'plunged', 'plunging',
    'underperform', 'underperformed', 'drag', 'dragged', 'drags', 'fear', 'fears',
    'worry', 'worries', 'concern', 'concerns', 'slow', 'slowing', 'slowdown', 'weak',
    'weaker', 'drop-off', 'inflationary', 'recession', 'panic', 'selloff'
}

def analyze_sentiment_text(text: str) -> dict:
    """
    Analyzes sentiment of a headline using a financial keyword dictionary.
    Returns: {"label": str, "score_discrete": float, "score_weighted": float,
              "prob_positive": float, "prob_negative": float}
    """
    words = re.findall(r'\b\w+\b', text.lower())
    
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    total = pos_count + neg_count
    
    if total > 0:
        score_weighted = (pos_count - neg_count) / total
        prob_positive = pos_count / (total + 2.0)
        prob_negative = neg_count / (total + 2.0)
    else:
        score_weighted = 0.0
        prob_positive = 0.0
        prob_negative = 0.0
        
    if score_weighted > 0.05:
        best_label = "positive"
        numeric = 1.0
    elif score_weighted < -0.05:
        best_label = "negative"
        numeric = -1.0
    else:
        best_label = "neutral"
        numeric = 0.0
        
    return {
        "label": best_label,
        "score_discrete": numeric,
        "score_weighted": round(score_weighted, 6),
        "prob_positive": round(prob_positive, 6),
        "prob_negative": round(prob_negative, 6)
    }


def score_headlines(
    headlines: list[str],
    *args,
    **kwargs
) -> list[dict]:
    """
    Runs lightweight sentiment over a list of headline strings.
    Accepts arbitrary arguments for backwards compatibility.
    """
    return [analyze_sentiment_text(h) for h in headlines]


# ---------------------------------------------------------------------------
# MODULE 4 — DAILY AGGREGATION
# ---------------------------------------------------------------------------

def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies lightweight sentiment to all fetched headlines and aggregates results by
    calendar date into a single sentiment score per day.

    Aggregation method
    ------------------
    - sentiment_score     : mean weighted score (pos_prob - neg_prob) per day
    - sentiment_discrete  : mean of discrete labels (+1 / 0 / -1) per day
    - headline_count      : number of articles processed that day
    - positive_ratio      : fraction of positive headlines
    - negative_ratio      : fraction of negative headlines
    - neutral_ratio       : fraction of neutral headlines
    - sentiment_std       : intraday standard deviation (uncertainty proxy)

    Returns
    -------
    pd.DataFrame indexed by date (DatetimeIndex, freq='D')
    """
    if news_df.empty:
        print("[WARN] No headlines to score. Returning empty DataFrame.")
        return pd.DataFrame()

    headlines = news_df["headline"].tolist()
    print(f"\n[Lightweight Sentiment] Scoring {len(headlines)} headlines...")
    scored = score_headlines(headlines)
    print(f"[Lightweight Sentiment] Inference complete.")

    # Attach scores back to news_df
    score_df = pd.DataFrame(scored)
    news_df  = news_df.reset_index(drop=True).join(score_df)

    # Ensure date column is datetime
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()

    # Group by day and aggregate
    agg = news_df.groupby("date").agg(
        sentiment_score    = ("score_weighted",  "mean"),
        sentiment_discrete = ("score_discrete",  "mean"),
        headline_count     = ("headline",        "count"),
        positive_ratio     = ("label",           lambda x: (x == "positive").mean()),
        negative_ratio     = ("label",           lambda x: (x == "negative").mean()),
        neutral_ratio      = ("label",           lambda x: (x == "neutral").mean()),
        sentiment_std      = ("score_weighted",  "std"),
    ).reset_index()

    agg["sentiment_score"]     = agg["sentiment_score"].round(6)
    agg["sentiment_discrete"]  = agg["sentiment_discrete"].round(4)
    agg["sentiment_std"]       = agg["sentiment_std"].fillna(0.0).round(6)
    agg["positive_ratio"]      = agg["positive_ratio"].round(4)
    agg["negative_ratio"]      = agg["negative_ratio"].round(4)
    agg["neutral_ratio"]       = agg["neutral_ratio"].round(4)

    agg.set_index("date", inplace=True)
    agg.index = pd.DatetimeIndex(agg.index)
    agg.sort_index(inplace=True)

    print(f"\n  [Aggregation] Daily sentiment rows: {len(agg)}")
    if len(agg) > 0:
        print(f"  [Aggregation] Date range: {agg.index[0].date()} -> {agg.index[-1].date()}")
        print(f"  [Aggregation] Overall mean score: {agg['sentiment_score'].mean():.4f}\n")

    return agg



# ---------------------------------------------------------------------------
# MODULE 5 — MERGER WITH SILVER DATASET
# ---------------------------------------------------------------------------

def merge_with_silver_dataset(
    sentiment_df: pd.DataFrame,
    silver_csv_path: str = "silver_ML_dataset.csv",
    output_csv_path: str = "silver_sentiment_ML_dataset.csv",
    fill_missing_sentiment: float = 0.0,
) -> pd.DataFrame:
    """
    Left-joins the sentiment DataFrame onto the existing silver OHLCV+features
    dataset by date. Trading days with no news get fill_missing_sentiment (0.0).

    Parameters
    ----------
    sentiment_df            : pd.DataFrame — output of aggregate_daily_sentiment()
    silver_csv_path         : str          — path to silver_ML_dataset.csv
    output_csv_path         : str          — where to save the merged dataset
    fill_missing_sentiment  : float        — default sentiment for no-news days

    Returns
    -------
    pd.DataFrame — merged, forward-filled ML-ready dataset
    """
    print(f"\n{'='*62}")
    print(f"  MERGER — Silver OHLCV + Sentiment Features")
    print(f"{'='*62}")

    silver_df = pd.read_csv(silver_csv_path, index_col=0, parse_dates=True)
    silver_df.index = pd.DatetimeIndex(silver_df.index).normalize()
    print(f"  [Silver] Loaded: {silver_df.shape[0]:,} rows x {silver_df.shape[1]} cols")

    if sentiment_df.empty:
        print("  [WARN] Sentiment data is empty. Adding zero-filled columns.")
        for col in ["sentiment_score", "sentiment_discrete", "headline_count",
                    "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std"]:
            silver_df[col] = fill_missing_sentiment
        merged = silver_df
    else:
        sentiment_df.index = pd.DatetimeIndex(sentiment_df.index).normalize()
        merged = silver_df.join(sentiment_df, how="left")

        # Fill trading days with no news coverage
        sentiment_cols = ["sentiment_score", "sentiment_discrete",
                          "positive_ratio", "negative_ratio", "neutral_ratio",
                          "sentiment_std"]
        merged[sentiment_cols] = merged[sentiment_cols].fillna(fill_missing_sentiment)
        merged["headline_count"] = merged["headline_count"].fillna(0).astype(int)

    merged.sort_index(inplace=True)
    merged.to_csv(output_csv_path)

    overlap = sentiment_df.index.isin(silver_df.index).sum() if not sentiment_df.empty else 0
    print(f"  [Merge] Matched {overlap} sentiment days to trading days.")
    print(f"  [Merge] Final shape: {merged.shape[0]:,} rows x {merged.shape[1]} cols")
    print(f"  [SAVE]  Saved -> '{output_csv_path}'\n")

    return merged


# ---------------------------------------------------------------------------
# MODULE 6 — SUMMARY PRINTER
# ---------------------------------------------------------------------------

def print_sentiment_summary(sentiment_df: pd.DataFrame) -> None:
    """Prints a readable summary of the daily sentiment scores."""
    if sentiment_df.empty:
        print("[INFO] No sentiment data to summarise.")
        return

    print(f"\n{'='*62}")
    print(f"  SENTIMENT SUMMARY  ({len(sentiment_df)} trading days)")
    print(f"{'='*62}")
    print(f"  Mean score      : {sentiment_df['sentiment_score'].mean():+.4f}")
    print(f"  Max score       : {sentiment_df['sentiment_score'].max():+.4f}")
    print(f"  Min score       : {sentiment_df['sentiment_score'].min():+.4f}")
    print(f"  Std deviation   : {sentiment_df['sentiment_score'].std():.4f}")
    print(f"  Avg headlines/d : {sentiment_df['headline_count'].mean():.1f}")
    print(f"  Total headlines : {sentiment_df['headline_count'].sum()}")

    # Counts
    n_pos = (sentiment_df["sentiment_score"] >  0.05).sum()
    n_neg = (sentiment_df["sentiment_score"] < -0.05).sum()
    n_neu = len(sentiment_df) - n_pos - n_neg
    print(f"\n  Days bullish  (score > +0.05) : {n_pos}")
    print(f"  Days bearish  (score < -0.05) : {n_neg}")
    print(f"  Days neutral                  : {n_neu}")

    print(f"\n  -- Recent 10 Days --")
    cols = ["sentiment_score", "headline_count", "positive_ratio", "negative_ratio"]
    print(sentiment_df[cols].tail(10).to_string())
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# MAIN PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_sentiment_pipeline(
    lookback_days: int = NEWS_LOOKBACK_DAYS,
    silver_csv: str    = "silver_ML_dataset.csv",
    output_csv: str    = "silver_sentiment_ML_dataset.csv",
    save_sentiment_csv: bool = True,
) -> dict:
    """
    End-to-end sentiment pipeline:
      1. Fetch all news (RSS + yfinance)
      2. Score with FinBERT
      3. Aggregate by day
      4. Merge with silver_ML_dataset.csv
      5. Save outputs

    Parameters
    ----------
    lookback_days       : int  — how many days back to collect news
    silver_csv          : str  — path to the silver OHLCV dataset
    output_csv          : str  — path for the merged output CSV
    save_sentiment_csv  : bool — also save raw sentiment CSV separately

    Returns
    -------
    dict:
        'news_df'      : raw collected headlines DataFrame
        'sentiment_df' : daily aggregated sentiment DataFrame
        'merged_df'    : merged silver + sentiment DataFrame
    """
    # Step 1 — Fetch
    news_df = fetch_all_news(lookback_days)

    # Step 2 & 3 — Score + Aggregate
    sentiment_df = aggregate_daily_sentiment(news_df)

    # Step 4 — Summary
    print_sentiment_summary(sentiment_df)

    # Step 5 — Optional raw sentiment save
    if save_sentiment_csv and not sentiment_df.empty:
        raw_path = "silver_daily_sentiment.csv"
        sentiment_df.to_csv(raw_path)
        print(f"  [SAVE] Raw sentiment -> '{raw_path}'")

    # Step 6 — Merge
    merged_df = merge_with_silver_dataset(sentiment_df, silver_csv, output_csv)

    return {
        "news_df"     : news_df,
        "sentiment_df": sentiment_df,
        "merged_df"   : merged_df,
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_sentiment_pipeline(
        lookback_days = NEWS_LOOKBACK_DAYS,
        silver_csv    = "silver_ML_dataset.csv",
        output_csv    = "silver_sentiment_ML_dataset.csv",
    )

    news_df      = results["news_df"]
    sentiment_df = results["sentiment_df"]
    merged_df    = results["merged_df"]

    print("[OK] Pipeline complete. Variables ready:")
    print("     news_df       -> Raw headlines DataFrame")
    print("     sentiment_df  -> Daily FinBERT sentiment scores")
    print("     merged_df     -> Full ML dataset (OHLCV + features + sentiment)")
    print()
    print("[USAGE] Import into your forecasting model:")
    print("   from news_sentiment_pipeline import run_sentiment_pipeline")
    print("   results    = run_sentiment_pipeline()")
    print("   merged_df  = results['merged_df']   # drop-in replacement for silver_ML_dataset.csv")
