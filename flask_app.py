"""
=============================================================================
 FLASK BACKEND — SILVER AI TRADING TERMINAL
=============================================================================
 Author  : Auto-generated for Advanced AI Time-Series Forecasting
 Purpose : Serves plain HTML dashboard, runs multi-step deep learning forecasts,
           features a Kurdish news sentiment advisor API (OpenRouter + Tavily),
           persists chat logs, and hosts a Telegram monitoring alert thread.
=============================================================================
"""

import os
import sys
import json
import time
import pickle
import warnings
import threading
import requests
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import telebot

# Try importing torch/PyTorch model
try:
    import torch
    import torch.nn as nn
    from pytorch_multimodal_model import SilverPredictor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

from openai import OpenAI
from tavily import TavilyClient

app = Flask(__name__)

# Ensure Flask json encoder returns raw UTF-8 string for Kurdish characters
app.json.ensure_ascii = False

# Import model structures and pipelining
from silver_data_pipeline import fetch_historical_data, clean_and_engineer
from news_sentiment_pipeline import fetch_all_news, aggregate_daily_sentiment



# ---------------------------------------------------------------------------
# INITIALIZATION & MODEL CACHING
# ---------------------------------------------------------------------------

MODEL_PATH = "silver_multimodal_predictor.pth"
SCALERS_PATH = "silver_scalers.pkl"
CHAT_HISTORY_FILE = "chat_history.json"
PORTFOLIO_FILE = "portfolio.json"

_cached_model = None
_cached_device = None
def get_model_and_device(ts_input_dim):
    global _cached_model, _cached_device
    if not HAS_TORCH:
        return None, None
    if _cached_model is None:
        _cached_device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SilverPredictor(
            ts_input_dim=ts_input_dim,
            lstm_hidden_dim=64,
            lstm_layers=2,
            sentiment_dim=1,
            sentiment_hidden_dim=16,
            fc_hidden_dim=32,
            dropout=0.2
        )
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=_cached_device))
        model.to(_cached_device)
        model.eval()
        _cached_model = model
    return _cached_model, _cached_device

# ---------------------------------------------------------------------------
# CHAT PERSISTENCE HELPERS
# ---------------------------------------------------------------------------

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[CHAT] Error loading chat history: {e}")
    return []


def save_chat_history(history):
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[CHAT] Error saving chat history: {e}")


# ---------------------------------------------------------------------------
# PAPER TRADING PORTFOLIO HELPERS
# ---------------------------------------------------------------------------

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        initial = {"cash_balance": 100000.00, "silver_oz_owned": 0.0}
        save_portfolio(initial)
        return initial
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "cash_balance" not in data:
                data["cash_balance"] = 100000.00
            if "silver_oz_owned" not in data:
                data["silver_oz_owned"] = 0.0
            return data
    except Exception as e:
        print(f"[PORTFOLIO] Error loading portfolio: {e}")
        return {"cash_balance": 100000.00, "silver_oz_owned": 0.0}

def save_portfolio(portfolio):
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(portfolio, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[PORTFOLIO] Error saving portfolio: {e}")


# ---------------------------------------------------------------------------
# TELEGRAM MONITOR ALERT BACKGROUND THREAD
# ---------------------------------------------------------------------------

def telegram_monitor_thread():
    """
    Background worker thread that monitors the silver market and sends Telegram alerts.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id or "YOUR_TELEGRAM" in bot_token or "YOUR_TELEGRAM" in chat_id:
        print("[TELEGRAM] Telegram Bot Token or Chat ID not configured in .env. Background alerts thread is on STANDBY.")
        return
        
    print(f"[TELEGRAM] Starting background monitor thread. Alerts will be sent to Chat ID: {chat_id}")
    last_sent_alert_date = None
    
    while True:
        try:
            # Check market conditions and send an alert once a day
            today_date = date.today().strftime("%Y-%m-%d")
            
            if last_sent_alert_date != today_date:
                print("[TELEGRAM] Running automated daily forecast evaluation...")
                forecast = calculate_multistep_forecast(horizon=1)
                latest_close = forecast["latest_close"]
                predicted_troy_oz = forecast["predicted_troy_oz"]
                pct_change = forecast["pct_change"]
                sentiment = forecast["sentiment_score"]
                
                # Format Kurdish Telegram message
                direction = "📈 بەرزبوونەوە" if pct_change >= 0 else "📉 دابەزین"
                color_emoji = "🟢" if pct_change >= 0 else "🔴"
                
                message = (
                    f"🔔 *ئاگادارکردنەوەی ڕۆژانەی بازاڕی زیو*\n\n"
                    f"📊 *نرخی ئێستا:* ${latest_close:.4f} / Troy Oz\n"
                    f"🔮 *پێشبینی سبەی (AI):* ${predicted_troy_oz:.4f} / Troy Oz\n"
                    f"📈 *ڕێژەی گۆڕانکاری پێشبینیکراو:* {color_emoji} {pct_change:+.2f}%\n"
                    f"💭 *هەستی بازاڕ (FinBERT):* {sentiment:+.4f}\n\n"
                    f"🏪 *ڕاوێژکاری زیوی سلێمانی (قەیسەری)*"
                )
                
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "Markdown"
                }
                res = requests.post(url, json=payload, timeout=15)
                
                if res.status_code == 200:
                    print(f"[TELEGRAM] Daily market alert sent successfully.")
                    last_sent_alert_date = today_date
                else:
                    print(f"[TELEGRAM] Failed to send Telegram alert: {res.text}")
                    
        except Exception as e:
            print(f"[TELEGRAM] Error in background monitor thread: {e}")
            
        # Sleep for 1 hour before evaluating again
        time.sleep(3600)


# ---------------------------------------------------------------------------
# FORECAST ROUTINES
# ---------------------------------------------------------------------------
def calculate_multistep_forecast(horizon=100, seq_len=20):
    """
    Fetches live data, runs multi-step prediction (either using PyTorch model or
    falling back to a lightweight trend+sentiment estimator), and packages data.
    """
    # 1. Fetch recent daily market data
    today_str = date.today().strftime("%Y-%m-%d")
    raw_history = fetch_historical_data(start="2026-02-01", end=today_str)
    cleaned_history = clean_and_engineer(raw_history)

    exclude_cols = [
        "sentiment_score", "sentiment_discrete", "headline_count",
        "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std"
    ]
    ts_cols = [col for col in cleaned_history.columns if col not in exclude_cols]

    # Get recent sequence slice
    recent_ts = cleaned_history[ts_cols].tail(seq_len)
    latest_close = float(cleaned_history["close"].iloc[-1])

    # 2. Get news sentiment today
    today_sentiment = 0.0
    sentiment_details = {"positive_ratio": 0.0, "negative_ratio": 0.0, "neutral_ratio": 1.0, "headline_count": 0}
    try:
        news_df = fetch_all_news(lookback_days=1)
        if not news_df.empty:
            sentiment_df = aggregate_daily_sentiment(news_df)
            if not sentiment_df.empty:
                today_sentiment = float(sentiment_df["sentiment_score"].iloc[-1])
                sentiment_details = {
                    "positive_ratio": float(sentiment_df["positive_ratio"].iloc[-1]),
                    "negative_ratio": float(sentiment_df["negative_ratio"].iloc[-1]),
                    "neutral_ratio": float(sentiment_df["neutral_ratio"].iloc[-1]),
                    "headline_count": int(sentiment_df["headline_count"].iloc[-1])
                }
        else:
            if os.path.exists("silver_daily_sentiment.csv"):
                saved = pd.read_csv("silver_daily_sentiment.csv", index_col=0)
                today_sentiment = float(saved["sentiment_score"].iloc[-1])
                sentiment_details = {
                    "positive_ratio": float(saved["positive_ratio"].iloc[-1]),
                    "negative_ratio": float(saved["negative_ratio"].iloc[-1]),
                    "neutral_ratio": float(saved["neutral_ratio"].iloc[-1]),
                    "headline_count": int(saved["headline_count"].iloc[-1])
                }
    except Exception:
        pass

    predictions = []
    
    # 3. Model Prediction or Fallback
    if HAS_TORCH and os.path.exists(SCALERS_PATH) and os.path.exists(MODEL_PATH):
        try:
            with open(SCALERS_PATH, "rb") as f:
                scalers = pickle.load(f)
            model, device = get_model_and_device(len(ts_cols))
            
            # Pre-scale sentiment
            scaler_sent = scalers["sent"]
            scaled_sent = scaler_sent.transform([[today_sentiment]])[0][0]
            sent_tensor = torch.tensor([[scaled_sent]], dtype=torch.float32).to(device)

            current_history = recent_ts.copy().values
            scaler_ts = scalers["ts"]
            scaler_target = scalers["target"]

            for step in range(horizon):
                # Scale current window
                scaled_ts = scaler_ts.transform(pd.DataFrame(current_history, columns=ts_cols))
                ts_tensor = torch.tensor(scaled_ts, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    norm_pred = model(ts_tensor, sent_tensor).cpu().numpy()

                pred_val = float(scaler_target.inverse_transform(norm_pred)[0][0])
                predictions.append(pred_val)

                # Update input array
                next_row = current_history[-1].copy()
                close_idx = ts_cols.index("close")
                next_row[close_idx] = pred_val
                current_history = np.vstack([current_history[1:], next_row])
        except Exception as e:
            print(f"[FORECAST] Error in PyTorch forecasting: {e}. Falling back to trend model.")
            predictions = []
            
    if not predictions:
        # Fallback: simple trend and sentiment forecasting using numpy/pandas
        current_close = latest_close
        
        # Calculate recent average daily return over last 5 trading days
        if len(cleaned_history) >= 5:
            recent_closes = cleaned_history["close"].tail(5).values
            pct_changes = (recent_closes[1:] - recent_closes[:-1]) / recent_closes[:-1]
            trend = float(np.mean(pct_changes))
        else:
            trend = 0.0001
            
        trend = max(-0.01, min(0.01, trend))  # Cap trend to ±1% per day
        
        # Enforce exactly 100-day future projection path
        for step in range(100):
            change_pct = trend + 0.002 * today_sentiment
            pred_val = current_close * (1.0 + change_pct)
            predictions.append(pred_val)
            current_close = pred_val

    # 6. Generate forecast date labels
    hist_dates = [d.strftime("%Y-%m-%d") for d in cleaned_history.index[-30:]]
    hist_prices = [float(p) for p in cleaned_history["close"].tail(30).values]

    forecast_dates = []
    last_date = cleaned_history.index[-1]
    if hasattr(last_date, "date"):
        current_date = last_date.date()
    elif isinstance(last_date, str):
        current_date = datetime.strptime(last_date, "%Y-%m-%d").date()
    else:
        current_date = last_date

    for _ in range(len(predictions)):
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5: # Shift weekends to trading days
            current_date += timedelta(days=1)
        forecast_dates.append(current_date.strftime("%Y-%m-%d"))

    # Convert prices
    final_pred = predictions[-1]
    pred_gram = final_pred / 31.103
    pred_kg = pred_gram * 1000
    pct_change = ((final_pred - latest_close) / latest_close) * 100

    # Generate rationale
    if pct_change > 0:
        if today_sentiment > 0.05:
            rationale = f"The algorithm projects an upward trend of {pct_change:.1f}% over the next 100 days driven by bullish momentum and positive news sentiment (score: {today_sentiment:.2f})."
        else:
            rationale = f"The algorithm projects a moderate growth of {pct_change:.1f}% over the next 100 days, primarily supported by steady historical price trends."
    else:
        if today_sentiment < -0.05:
            rationale = f"The algorithm projects a downward trend of {pct_change:.1f}% over the next 100 days driven by bearish momentum and negative news sentiment (score: {today_sentiment:.2f})."
        else:
            rationale = f"The algorithm projects a slight decline of {pct_change:.1f}% over the next 100 days, reflecting recent consolidations in the historical trend."

    return {
        "predicted_troy_oz": final_pred,
        "predicted_gram": pred_gram,
        "predicted_kg": pred_kg,
        "latest_close": latest_close,
        "pct_change": pct_change,
        "sentiment_score": today_sentiment,
        "sentiment_details": sentiment_details,
        "historical_dates": hist_dates,
        "historical_prices": hist_prices,
        "forecast_dates": forecast_dates,
        "forecast_prices": predictions,
        "rationale": rationale
    }


# ---------------------------------------------------------------------------
# APP ENDPOINTS
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    portfolio = load_portfolio()
    return render_template("index.html", portfolio=portfolio)


@app.route("/api/portfolio", methods=["GET"])
def portfolio_endpoint():
    return jsonify(load_portfolio())


@app.route("/api/trade", methods=["POST"])
def trade_endpoint():
    try:
        data = request.get_json() or {}
        action = data.get("action", "").lower()
        
        try:
            amount_oz = float(data.get("amount_oz", 0))
        except (ValueError, TypeError):
            return jsonify({"error": "بڕی دیاریکراو دەبێت ژمارە بێت"}), 400
            
        if action not in ["buy", "sell"]:
            return jsonify({"error": "کرداری نادیار (کڕین یان فرۆشتن)"}), 400
        if amount_oz <= 0:
            return jsonify({"error": "بڕی زیو دەبێت لە 0 زیاتر بێت"}), 400
            
        # Get live Silver price
        forecast = calculate_multistep_forecast(horizon=1)
        current_price = float(forecast["latest_close"])
        
        portfolio = load_portfolio()
        cash = portfolio["cash_balance"]
        holdings = portfolio["silver_oz_owned"]
        
        total_cost = current_price * amount_oz
        
        if action == "buy":
            if total_cost > cash:
                return jsonify({"error": f"کاشی پێویستت نییە. تێچووی گشتی: ${total_cost:.2f}، بەڵام کاشی بەردەستت: ${cash:.2f}"}), 400
            portfolio["cash_balance"] = round(cash - total_cost, 2)
            portfolio["silver_oz_owned"] = round(holdings + amount_oz, 4)
            msg = f"کڕینی {amount_oz:.4f} ئۆنس زیو بە سەرکەوتوویی ئەنجامدرا بە نرخی ${current_price:.4f} بۆ هەر ئۆنسێک."
        else: # sell
            if amount_oz > holdings:
                return jsonify({"error": f"بڕی زیوی پێویستت نییە بۆ فرۆشتن. بڕی بەردەست: {holdings:.4f} Oz"}), 400
            portfolio["cash_balance"] = round(cash + total_cost, 2)
            portfolio["silver_oz_owned"] = round(holdings - amount_oz, 4)
            msg = f"فرۆشتنی {amount_oz:.4f} ئۆنس زیو بە سەرکەوتوویی ئەنجامدرا بە نرخی ${current_price:.4f} بۆ هەر ئۆنسێک."
            
        save_portfolio(portfolio)
        return jsonify({
            "message": msg,
            "cash_balance": portfolio["cash_balance"],
            "silver_oz_owned": portfolio["silver_oz_owned"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/predict")
def predict_endpoint():
    try:
        horizon = int(request.args.get("horizon", 100))
        horizon = max(1, min(100, horizon))
        forecast_results = calculate_multistep_forecast(horizon=horizon)
        return jsonify(forecast_results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/history", methods=["GET"])
def chat_history_endpoint():
    return jsonify(load_chat_history())


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "پرسیارەکە بەتاڵە"}), 400

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        if not openrouter_key:
            return jsonify({"error": "کلیلەکانی OpenRouter API لەسەر سێرڤەر دانەمەزراون."}), 500

        # Load existing history
        history = load_chat_history()

        # 1. Search Tavily for news
        search_results = ""
        try:
            tavily = TavilyClient(api_key=tavily_key)
            query = "silver market news economic conditions Iraq Kurdistan Sulaymaniyah"
            search_response = tavily.search(query=query, max_results=3)
            search_results = "\n\n".join([
                f"Title: {r['title']}\nSnippet: {r['content']}" 
                for r in search_response.get('results', [])
            ])
        except Exception as te:
            print(f"[WARN] Tavily search failed: {te}. Proceeding without search context.")
            search_results = "No recent external search results available due to API access limits."

        # 2. Get latest forecasting values for context
        forecast = calculate_multistep_forecast(horizon=1)
        model_telemetry = (
            f"Silver Market Telemetry:\n"
            f"- Today's Last Actual Close: ${forecast['latest_close']:.4f} / Troy Oz\n"
            f"- Tomorrow's AI Predicted Price: ${forecast['predicted_troy_oz']:.4f} / Troy Oz\n"
            f"- Tomorrow's Predicted Price (Kg): ${forecast['predicted_kg']:.2f}\n"
            f"- Sentiment Score Indicator: {forecast['sentiment_score']:+.4f}"
        )

        # 3. Call OpenRouter using OpenAI client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )
        system_prompt = (
            "You are an expert Silver Market Advisor located in the Qaysari Bazaar of Sulaymaniyah, Kurdistan. "
            "You MUST speak fluent Kurdish (Sorani). Analyze the data and advise the user. Be concise and professional."
        )
        user_payload = (
            f"User Question: {user_message}\n\n"
            f"{model_telemetry}\n\n"
            f"Latest Economic & Silver News:\n{search_results}"
        )

        chat_completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "Silver AI Terminal",
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload}
            ],
            model="meta-llama/llama-3.1-70b-instruct",
            temperature=0.3
        )
        response_text = chat_completion.choices[0].message.content

        # Append to persisted chat history
        history.append({"role": "user", "text": user_message})
        history.append({"role": "advisor", "text": response_text})
        save_chat_history(history)

        return jsonify({"response": response_text})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# TELEGRAM BOT COMMAND HANDLERS & POLLING
# ---------------------------------------------------------------------------

bot = None
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

if bot_token and "YOUR_TELEGRAM" not in bot_token:
    try:
        bot = telebot.TeleBot(bot_token)
    except Exception as e:
        print(f"[TELEGRAM] Error initializing TeleBot: {e}")

if bot:
    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        welcome_msg = (
            "👋 بەخێربێیت بۆ بۆتی فەرمی بازاڕی زیوی سلێمانی!\n\n"
            "من ڕاوێژکاری زیرەکی تۆم بۆ پێشبینیکردن و شیکردنەوەی بازاڕی زیو.\n\n"
            "📌 فەرمانەکان:\n"
            "/predict - پێشبینی نرخی زیو بۆ سبەی\n"
            "/help - نیشاندانی ئەم نامەیە\n\n"
            "هەروەها دەتوانیت هەر پرسیارێکی ترت هەیە بە کوردی لێم بپرسیت!"
        )
        bot.reply_to(message, welcome_msg)

    @bot.message_handler(commands=['predict'])
    def send_prediction(message):
        try:
            forecast = calculate_multistep_forecast(horizon=1)
            latest_close = forecast["latest_close"]
            predicted_troy_oz = forecast["predicted_troy_oz"]
            pct_change = forecast["pct_change"]
            sentiment = forecast["sentiment_score"]
            
            direction = "📈 بەرزبوونەوە" if pct_change >= 0 else "📉 دابەزین"
            color_emoji = "🟢" if pct_change >= 0 else "🔴"
            
            msg = (
                f"🔮 *پێشبینی بازاڕی زیو بۆ سبەی:*\n\n"
                f"📊 *نرخی ئێستا:* ${latest_close:.4f} / Troy Oz\n"
                f"🔮 *نرخی پێشبینیکراو:* ${predicted_troy_oz:.4f} / Troy Oz\n"
                f"📈 *ڕێژەی گۆڕانکاری:* {color_emoji} {pct_change:+.2f}%\n"
                f"💭 *هەستی بازاڕ:* {sentiment:+.4f}\n"
            )
            bot.reply_to(message, msg, parse_mode="Markdown")
        except Exception as e:
            bot.reply_to(message, f"❌ ببورە، کێشەیەک لە پێشبینیکردندا ڕوویدا: {str(e)}")

    @bot.message_handler(func=lambda message: True)
    def handle_kurdish_chat(message):
        user_message = message.text.strip()
        if not user_message:
            return
            
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        if not openrouter_key:
            bot.reply_to(message, "❌ ببورە، کلیلەکانی API لەسەر سێرڤەر دانەمەزراون.")
            return

        try:
            bot.send_chat_action(message.chat.id, 'typing')
            
            search_results = ""
            try:
                tavily = TavilyClient(api_key=tavily_key)
                query = f"silver market news {user_message}"
                search_response = tavily.search(query=query, max_results=2)
                search_results = "\n\n".join([
                    f"Title: {r['title']}\nSnippet: {r['content']}" 
                    for r in search_response.get('results', [])
                ])
            except Exception as te:
                print(f"[WARN] Tavily search failed in Telegram: {te}")
                search_results = "No recent external search results available."

            forecast = calculate_multistep_forecast(horizon=1)
            model_telemetry = (
                f"Silver Market Telemetry:\n"
                f"- Today's Last Actual Close: ${forecast['latest_close']:.4f} / Troy Oz\n"
                f"- Tomorrow's AI Predicted Price: ${forecast['predicted_troy_oz']:.4f} / Troy Oz\n"
                f"- Sentiment Score: {forecast['sentiment_score']:+.4f}"
            )

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )
            system_prompt = (
                "You are an expert Silver Market Advisor located in the Qaysari Bazaar of Sulaymaniyah, Kurdistan. "
                "You MUST speak fluent Kurdish (Sorani). Analyze the data and reply to the user. Keep it concise, friendly, and under 150 words."
            )
            user_payload = (
                f"User Question: {user_message}\n\n"
                f"{model_telemetry}\n\n"
                f"Latest News:\n{search_results}"
            )

            chat_completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost:5000",
                    "X-Title": "Silver AI Telegram Bot",
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload}
                ],
                model="meta-llama/llama-3.1-70b-instruct",
                temperature=0.3
            )
            response_text = chat_completion.choices[0].message.content
            bot.reply_to(message, response_text)
        except Exception as e:
            bot.reply_to(message, f"❌ ببورە، هەڵەیەک ڕوویدا لە وەڵامدانەوەدا: {str(e)}")


def telegram_polling_thread():
    """
    Background worker thread that runs the Telegram bot polling.
    """
    if bot:
        print("[TELEGRAM] Checking Bot authentication...")
        try:
            bot.get_me()
            print("[TELEGRAM] Bot authentication successful.")
        except Exception as e:
            print(f"[TELEGRAM] Bot unauthorized or offline, disabling polling: {e}")
            return

        print("[TELEGRAM] Starting bot polling thread...")
        try:
            bot.infinity_polling()
        except Exception as e:
            print(f"[TELEGRAM] Error in bot polling: {e}")


def start_telegram_threads():
    """
    Starts background threads for daily alerts and polling, ensuring they only run once.
    """
    global _threads_started
    if '_threads_started' not in globals():
        globals()['_threads_started'] = True
        
        # Start Daily alerts thread
        t1 = threading.Thread(target=telegram_monitor_thread, daemon=True)
        t1.start()
        
        # Start Live polling thread
        if bot:
            t2 = threading.Thread(target=telegram_polling_thread, daemon=True)
            t2.start()

# Automatically trigger the background threads when module is loaded (e.g. by Gunicorn)
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    start_telegram_threads()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

