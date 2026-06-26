"""
=============================================================================
 SILVER FUTURES (SI=F) — EVALUATION & INFERENCE PIPELINE
=============================================================================
 Author  : Auto-generated for Advanced AI Time-Series Forecasting
 Purpose : 1. Evaluates the Multi-modal PyTorch model on validation data.
           2. Visualizes Actual vs. Predicted Close prices in a high-res chart.
           3. Fetches live market data + today's news sentiment.
           4. Formulates next-day forecast and converts to Grams & Kilograms.
=============================================================================
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Import components from existing pipelines
from silver_data_pipeline import fetch_historical_data, clean_and_engineer
from news_sentiment_pipeline import fetch_all_news, aggregate_daily_sentiment
from pytorch_multimodal_model import SilverPredictor, prepare_data_loaders

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# PART 1 — EVALUATION & VISUALIZATION
# ---------------------------------------------------------------------------

def run_evaluation_and_plotting(
    model_path: str = "silver_multimodal_predictor.pth",
    scalers_path: str = "silver_scalers.pkl",
    merged_csv_path: str = "silver_sentiment_ML_dataset.csv",
    seq_len: int = 20,
    output_chart_path: str = "evaluation_chart.png"
):
    """
    Loads the trained model and validation data, performs evaluation,
    calculates metrics, and saves a professional actual vs. predicted plot.
    """
    print(f"\n{'='*62}")
    print(f"  PART 1 - MODEL EVALUATION & VISUALIZATION")
    print(f"{'='*62}")

    # Check dependencies
    for path in (model_path, scalers_path, merged_csv_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Required resource not found: {path}")

    # 1. Load Scalers
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    print(f"  [Load] Scalers loaded successfully from '{scalers_path}'")

    # 2. Load merged dataset
    df = pd.read_csv(merged_csv_path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index).normalize()
    print(f"  [Load] Dataset loaded from '{merged_csv_path}' ({len(df)} rows)")

    # 3. Setup features
    exclude_cols = [
        "sentiment_score", "sentiment_discrete", "headline_count",
        "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std"
    ]
    ts_cols = [col for col in df.columns if col not in exclude_cols]

    # 4. Prepare data loaders (with identical split parameters)
    _, val_loader, _, val_df, target_col = prepare_data_loaders(
        df=df,
        seq_len=seq_len,
        ts_features=ts_cols,
        sentiment_feature="sentiment_score",
        target_feature="close"
    )

    # 5. Instantiate and load model weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SilverPredictor(
        ts_input_dim=len(ts_cols),
        lstm_hidden_dim=64,
        lstm_layers=2,
        sentiment_dim=1,
        sentiment_hidden_dim=16,
        fc_hidden_dim=32,
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  [Load] PyTorch model loaded successfully from '{model_path}'")

    # 6. Forward pass on Validation data
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_ts, batch_sent, batch_y in val_loader:
            batch_ts, batch_sent = batch_ts.to(device), batch_sent.to(device)
            preds = model(batch_ts, batch_sent)
            all_preds.extend(preds.cpu().numpy())
            all_actuals.extend(batch_y.numpy())

    all_preds = np.array(all_preds).reshape(-1, 1)
    all_actuals = np.array(all_actuals).reshape(-1, 1)

    # 7. Inverse transform to original currency prices
    inv_preds = scalers["target"].inverse_transform(all_preds).flatten()
    inv_actuals = scalers["target"].inverse_transform(all_actuals).flatten()

    # Calculate metrics
    mae = np.mean(np.abs(inv_preds - inv_actuals))
    rmse = np.sqrt(np.mean((inv_preds - inv_actuals) ** 2))
    
    print(f"  [Eval] Mean Absolute Error (MAE): ${mae:.4f}")
    print(f"  [Eval] Root Mean Squared Error (RMSE): ${rmse:.4f}")

    # 8. Plot actual vs predicted prices
    plot_dates = val_df.index[seq_len:]
    # Handle possible length mismatch due to DataLoader drop_last
    plot_dates = plot_dates[:len(inv_preds)]

    plt.figure(figsize=(12, 6), dpi=300)
    plt.plot(plot_dates, inv_actuals, label="Actual Silver Price", color="#95a5a6", linewidth=2.0)
    plt.plot(plot_dates, inv_preds, label="Predicted Silver Price", color="#e74c3c", linestyle="--", linewidth=1.8)
    
    plt.title("Silver Futures (SI=F) Daily Forecast - Actual vs. Predicted Close Price", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Date", fontsize=11, labelpad=10)
    plt.ylabel("Price (USD / Troy Ounce)", fontsize=11, labelpad=10)
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#bdc3c7")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_chart_path)
    plt.close()
    print(f"  [OK] Saved high-resolution evaluation chart -> '{output_chart_path}'\n")


# ---------------------------------------------------------------------------
# PART 2 — LIVE INFERENCE & CONVERSION
# ---------------------------------------------------------------------------

def run_live_inference(
    model_path: str = "silver_multimodal_predictor.pth",
    scalers_path: str = "silver_scalers.pkl",
    seq_len: int = 20
):
    """
    Fetches latest live data + today's news sentiment, preprocesses it,
    predicts the next close price, and formats/prints the unit conversions.
    """
    print(f"{'='*62}")
    print(f"  PART 2 - LIVE INFERENCE & PRICE CONVERSIONS")
    print(f"{'='*62}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Scalers
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    # 2. Fetch recent daily market data to get the required history window (seq_len days)
    print("  [Live] Fetching recent market history via yfinance...")
    # Fetch 90 days to ensure we have enough trading days after weekend/holiday exclusions
    today_str = date.today().strftime("%Y-%m-%d")
    raw_history = fetch_historical_data(start="2026-03-01", end=today_str)
    cleaned_history = clean_and_engineer(raw_history)
    
    # Extract feature list
    exclude_cols = [
        "sentiment_score", "sentiment_discrete", "headline_count",
        "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std"
    ]
    ts_cols = [col for col in cleaned_history.columns if col not in exclude_cols]

    # Get latest seq_len days of time series features
    recent_ts = cleaned_history[ts_cols].tail(seq_len)
    if len(recent_ts) < seq_len:
        raise ValueError(f"[ERROR] Insufficient recent data: got {len(recent_ts)} rows, need {seq_len}")

    # 3. Fetch today's news headlines & run FinBERT sentiment pipeline
    print("  [Live] Fetching today's news and scoring sentiment...")
    today_sentiment = 0.0
    try:
        # Fetch today's news headlines
        news_df = fetch_all_news(lookback_days=1)
        if not news_df.empty:
            sentiment_df = aggregate_daily_sentiment(news_df)
            if not sentiment_df.empty:
                # Get average sentiment score of today
                today_sentiment = sentiment_df["sentiment_score"].iloc[-1]
                print(f"  [Live] Live Sentiment Score computed: {today_sentiment:+.4f}")
            else:
                print("  [WARN] Daily sentiment aggregation was empty. Defaulting to 0.0.")
        else:
            # Fall back to the last saved sentiment
            if os.path.exists("silver_daily_sentiment.csv"):
                saved_sent_df = pd.read_csv("silver_daily_sentiment.csv", index_col=0)
                today_sentiment = saved_sent_df["sentiment_score"].iloc[-1]
                print(f"  [Live] No new headlines today. Loaded last saved sentiment: {today_sentiment:+.4f}")
            else:
                print("  [WARN] No news fetched and no saved sentiment file found. Using neutral (0.0).")
    except Exception as e:
        print(f"  [WARN] Live sentiment fetch failed: {e}. Using neutral (0.0) fallback.")

    # 4. Preprocess inputs using fitted training scalers
    scaled_ts = scalers["ts"].transform(recent_ts)
    scaled_sent = scalers["sent"].transform([[today_sentiment]])[0][0]

    # Convert to PyTorch tensors
    ts_tensor = torch.tensor(scaled_ts, dtype=torch.float32).unsqueeze(0).to(device) # Shape: (1, seq_len, ts_dim)
    sent_tensor = torch.tensor([[scaled_sent]], dtype=torch.float32).to(device)     # Shape: (1, 1)

    # 5. Load model & perform forward pass
    model = SilverPredictor(
        ts_input_dim=len(ts_cols),
        lstm_hidden_dim=64,
        lstm_layers=2,
        sentiment_dim=1,
        sentiment_hidden_dim=16,
        fc_hidden_dim=32,
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        normalized_pred = model(ts_tensor, sent_tensor)
        normalized_pred = normalized_pred.cpu().numpy()

    # 6. Inverse-transform to get the dollar price
    predicted_troy_oz = scalers["target"].inverse_transform(normalized_pred)[0][0]

    # 7. Convert units
    troy_oz_to_gram = 31.103
    predicted_gram = predicted_troy_oz / troy_oz_to_gram
    predicted_kg = predicted_gram * 1000

    # 8. Beautiful formatted console output
    print(f"\n{'='*62}")
    print(f"  SILVER FUTURES MODEL PREDICTION FOR NEXT TRADING DAY")
    print(f"{'='*62}")
    print(f"  Live Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input Window Dates  : {recent_ts.index[0].date()} to {recent_ts.index[-1].date()}")
    print(f"  Live Sentiment Score: {today_sentiment:+.4f}")
    print(f"")
    print(f"  +---------------------------------------------+")
    print(f"  |  PREDICTED CLOSE PRICE                      |")
    print(f"  |                                             |")
    print(f"  |  Per Troy Ounce : USD {predicted_troy_oz:>14.4f}        |")
    print(f"  |  Per Gram       : USD {predicted_gram:>14.6f}        |")
    print(f"  |  Per Kilogram   : USD {predicted_kg:>14.4f}        |")
    print(f"  |                                             |")
    print(f"  |  Conversion: 1 Troy Oz = {troy_oz_to_gram} Grams         |")
    print(f"  +---------------------------------------------+")
    print(f"{'='*62}\n")
    return predicted_troy_oz, cleaned_history["close"].iloc[-1]


if __name__ == "__main__":
    run_evaluation_and_plotting()
    run_live_inference()
