"""
=============================================================================
 STREAMLIT APP — SILVER AI TRADING TERMINAL
=============================================================================
 Author  : Auto-generated for Advanced AI Time-Series Forecasting
 Purpose : Interactive web dashboard displaying multi-day forecast,
           live sentiment analytics, and trade execution status.
=============================================================================
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Silver AI Trading Terminal",
    page_icon="🪙",
    layout="wide"
)

# Import model structures
from silver_data_pipeline import fetch_historical_data, clean_and_engineer
from news_sentiment_pipeline import fetch_all_news, aggregate_daily_sentiment
from pytorch_multimodal_model import SilverPredictor


# ---------------------------------------------------------------------------
# MODEL LOADING & MULTI-DAY FORECAST ENGINE
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model(model_path, ts_input_dim):
    """Loads and caches the forecasting model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SilverPredictor(
        ts_input_dim=ts_input_dim,
        lstm_hidden_dim=64,
        lstm_layers=2,
        sentiment_dim=1,
        sentiment_hidden_dim=16,
        fc_hidden_dim=32,
        dropout=0.2
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_terminal_data():
    """Fetches and prepares all the market and sentiment data."""
    # 1. Load Scalers
    scalers_path = "silver_scalers.pkl"
    if not os.path.exists(scalers_path):
        return None, f"Scalers not found at: {scalers_path}"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    # 2. Fetch history
    today_str = date.today().strftime("%Y-%m-%d")
    raw_history = fetch_historical_data(start="2026-02-01", end=today_str)
    cleaned_history = clean_and_engineer(raw_history)

    exclude_cols = [
        "sentiment_score", "sentiment_discrete", "headline_count",
        "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std"
    ]
    ts_cols = [col for col in cleaned_history.columns if col not in exclude_cols]

    # 3. Fetch sentiment
    sentiment_score = 0.0
    sentiment_data = {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "count": 0}
    
    try:
        news_df = fetch_all_news(lookback_days=1)
        if not news_df.empty:
            sentiment_df = aggregate_daily_sentiment(news_df)
            if not sentiment_df.empty:
                sentiment_score = sentiment_df["sentiment_score"].iloc[-1]
                sentiment_data = {
                    "positive": sentiment_df["positive_ratio"].iloc[-1],
                    "negative": sentiment_df["negative_ratio"].iloc[-1],
                    "neutral": sentiment_df["neutral_ratio"].iloc[-1],
                    "count": int(sentiment_df["headline_count"].iloc[-1])
                }
        else:
            if os.path.exists("silver_daily_sentiment.csv"):
                saved = pd.read_csv("silver_daily_sentiment.csv", index_col=0)
                sentiment_score = saved["sentiment_score"].iloc[-1]
                sentiment_data = {
                    "positive": saved["positive_ratio"].iloc[-1],
                    "negative": saved["negative_ratio"].iloc[-1],
                    "neutral": saved["neutral_ratio"].iloc[-1],
                    "count": int(saved["headline_count"].iloc[-1])
                }
    except Exception:
        pass

    return {
        "scalers": scalers,
        "history": cleaned_history,
        "ts_cols": ts_cols,
        "sentiment_score": sentiment_score,
        "sentiment_data": sentiment_data
    }, None


def predict_multistep(data, horizon_days=1, seq_len=20):
    """
    Computes an auto-regressive multi-step forecast using the LSTM model.
    """
    scalers = data["scalers"]
    history = data["history"].copy()
    ts_cols = data["ts_cols"]
    sentiment_score = data["sentiment_score"]

    model_path = "silver_multimodal_predictor.pth"
    model, device = get_model(model_path, len(ts_cols))

    predictions = []
    current_history = history[ts_cols].tail(seq_len).values

    # Fit scaling
    scaler_ts = scalers["ts"]
    scaler_sent = scalers["sent"]
    scaler_target = scalers["target"]

    scaled_sent = scaler_sent.transform([[sentiment_score]])[0][0]
    sent_tensor = torch.tensor([[scaled_sent]], dtype=torch.float32).to(device)

    for step in range(horizon_days):
        # Scale time-series input
        scaled_ts = scaler_ts.transform(pd.DataFrame(current_history, columns=ts_cols))
        ts_tensor = torch.tensor(scaled_ts, dtype=torch.float32).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            norm_pred = model(ts_tensor, sent_tensor).cpu().numpy()

        pred_val = float(scaler_target.inverse_transform(norm_pred)[0][0])
        predictions.append(pred_val)

        # Auto-regressively roll forward: 
        # Create a new step row by shifting values and placing the predicted price in 'close'
        next_row = current_history[-1].copy()
        # Find index of close column
        close_idx = ts_cols.index("close")
        next_row[close_idx] = pred_val
        
        # Append and slide window
        current_history = np.vstack([current_history[1:], next_row])

    return predictions


# ---------------------------------------------------------------------------
# UI DRAWING
# ---------------------------------------------------------------------------

def draw_dashboard():
    # Style definitions
    st.markdown("""
        <style>
        .terminal-title {
            font-size: 2.8rem;
            font-weight: 800;
            color: #1a1a1a;
            margin-bottom: 2px;
        }
        .terminal-subtitle {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 30px;
        }
        .status-standby {
            background-color: #f39c12;
            color: white;
            padding: 4px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 0.85rem;
            display: inline-block;
        }
        .metric-card {
            background-color: #fcfcfc;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #eaeaea;
        }
        </style>
    """, unsafe_allowed_html=True)

    st.markdown('<div class="terminal-title">🪙 Silver AI Trading Terminal</div>', unsafe_allowed_html=True)
    st.markdown('<div class="terminal-subtitle">Deep learning prediction terminal & real-time sentiment analyzer</div>', unsafe_allowed_html=True)

    # 1. Fetch live system data
    with st.spinner("Fetching latest live data and scoring market sentiment..."):
        data, err = fetch_terminal_data()

    if err:
        st.error(err)
        return

    # 2. Sidebar Controls
    with st.sidebar:
        st.markdown("### 🎛️ Terminal Controls")
        
        # Date selector/Forecast Horizon slider
        horizon = st.slider(
            label="Forecast Horizon (Days)",
            min_value=1,
            max_value=10,
            value=1,
            help="Select how many steps/trading days into the future to forecast using auto-regressive prediction."
        )

        st.markdown("---")
        st.markdown("### 📊 Live News Sentiment")
        
        sent_val = data["sentiment_score"]
        # Bullish/Bearish metric with custom HTML coloring
        if sent_val > 0.05:
            st.markdown(f"Sentiment: <span style='color:#2ecc71; font-weight:bold; font-size:1.2rem;'>BULLISH ({sent_val:+.4f})</span>", unsafe_allowed_html=True)
        elif sent_val < -0.05:
            st.markdown(f"Sentiment: <span style='color:#e74c3c; font-weight:bold; font-size:1.2rem;'>BEARISH ({sent_val:+.4f})</span>", unsafe_allowed_html=True)
        else:
            st.markdown(f"Sentiment: <span style='color:#7f8c8d; font-weight:bold; font-size:1.2rem;'>NEUTRAL ({sent_val:+.4f})</span>", unsafe_allowed_html=True)

        st.markdown("---")
        st.markdown("### 🤖 Trading System Status")
        st.markdown("Bot Engine: <span class='status-standby'>STANDBY</span>", unsafe_allowed_html=True)

    # 3. Forecast calculations
    predictions = predict_multistep(data, horizon_days=horizon)
    target_pred = predictions[-1]
    latest_close = data["history"]["close"].iloc[-1]
    pct_change = ((target_pred - latest_close) / latest_close) * 100

    # Conversions
    troy_oz_to_gram = 31.103
    pred_gram = target_pred / troy_oz_to_gram
    pred_kg = pred_gram * 1000

    # 4. Top Metrics Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label=f"Predicted Price (1 Troy Ounce) — Day {horizon}",
            value=f"${target_pred:,.4f}",
            delta=f"{pct_change:+.2f}% vs. Today",
            delta_color="normal"
        )
    with col2:
        st.metric(
            label=f"Predicted Price (1 Gram) — Day {horizon}",
            value=f"${pred_gram:,.6f}",
            delta=f"Ratio: 31.103g/Oz"
        )
    with col3:
        st.metric(
            label=f"Predicted Price (1 Kilogram) — Day {horizon}",
            value=f"${pred_kg:,.4f}",
            delta=f"Ratio: 1000g/Kg"
        )

    # 5. Interactive Chart Plotting
    st.markdown("### 📈 Interactive Forecast Visualization")
    
    chart_df = data["history"][["close"]].tail(30)
    dates = list(chart_df.index)
    prices = list(chart_df["close"])

    # Create forecast dates
    forecast_dates = []
    current_date = dates[-1]
    for _ in range(horizon):
        current_date += timedelta(days=1)
        # Avoid weekends on line chart
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        forecast_dates.append(current_date)

    fig = go.Figure()

    # Historical Close
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode="lines+markers",
        name="Historical Price",
        line=dict(color="#2980b9", width=2.5),
        marker=dict(size=4),
        hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.4f}<extra></extra>"
    ))

    # Forecast steps
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions,
        mode="lines+markers",
        name="AI Forecast Path",
        line=dict(color="#e74c3c", width=2, dash="dash"),
        marker=dict(color="#e74c3c", size=7, symbol="diamond"),
        hovertemplate="<b>Day %{text}:</b> %{x}<br><b>Predicted Price:</b> $%{y:.4f}<extra></extra>",
        text=list(range(1, horizon + 1))
    ))

    # Highlight target horizon day with a large marker
    fig.add_trace(go.Scatter(
        x=[forecast_dates[-1]],
        y=[target_pred],
        mode="markers",
        name=f"Horizon Day {horizon} Prediction",
        marker=dict(color="#27ae60", size=14, symbol="star", line=dict(color="#219653", width=1.5)),
        hovertemplate="<b>Target Forecast:</b> %{x}<br><b>Predicted Price:</b> $%{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(showgrid=True, gridcolor="#f1f2f6"),
        yaxis=dict(showgrid=True, gridcolor="#f1f2f6", tickformat="$,.2f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    draw_dashboard()
