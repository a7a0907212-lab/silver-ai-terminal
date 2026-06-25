# -*- coding: utf-8 -*-
"""
=============================================================================
 COMPILATION INSTRUCTIONS FOR PYINSTALLER
=============================================================================
 Run the following command in the terminal to compile this script into a 
 standalone, folder-based Windows desktop executable:
 
 pyinstaller --noconfirm --onedir --windowed --add-data "C:\\Users\\MICROHEM\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\customtkinter;customtkinter/" silver_desktop_app.py
=============================================================================
"""

import os
import sys
import pickle
import threading
import warnings
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import torch
import customtkinter as ctk

# Matplotlib integration in Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Groq and Tavily APIs
from groq import Groq
from tavily import TavilyClient

# Force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Import models
from silver_data_pipeline import fetch_historical_data, clean_and_engineer
from pytorch_multimodal_model import SilverPredictor

# ---------------------------------------------------------------------------
# APPEARANCE CONFIGURATION
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class SilverDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Silver AI Trading Terminal")
        self.geometry("1400x820")
        self.grid_columnconfigure(0, weight=3) # Left Panel weight
        self.grid_columnconfigure(1, weight=2) # Right Panel weight
        self.grid_rowconfigure(0, weight=1)

        # State Variables
        self.predicted_troy_oz = 0.0
        self.predicted_kg = 0.0
        self.latest_close = 0.0
        self.sentiment_score = 0.0

        # Create panels
        self.create_left_panel()
        self.create_right_panel()

        # Load data and run prediction
        self.load_model_and_forecast()

    # ---------------------------------------------------------------------------
    # PANEL SETUP
    # ---------------------------------------------------------------------------

    def create_left_panel(self):
        """Creates the market telemetry and chart display panel."""
        self.left_frame = ctk.CTkFrame(self, corner_radius=15, border_width=1, border_color="#2c3e50")
        self.left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1) # Chart gets weight

        # Header Title
        title_label = ctk.CTkLabel(
            self.left_frame, 
            text="🥈 Live Market Telemetry & Forecast", 
            font=ctk.CTkFont(family="Outfit", size=22, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=15, sticky="w")

        # Telemetry Display Box
        self.telemetry_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.telemetry_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.telemetry_frame.grid_columnconfigure((0, 1), weight=1)

        # Card 1: Troy Ounce
        self.card_ounce = ctk.CTkFrame(self.telemetry_frame, corner_radius=10, fg_color="#1e272e")
        self.card_ounce.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.lbl_oz_title = ctk.CTkLabel(self.card_ounce, text="TROY OUNCE FORECAST", font=ctk.CTkFont(size=11, weight="bold"), text_color="#bdc3c7")
        self.lbl_oz_title.pack(pady=(12, 2))
        self.lbl_oz_val = ctk.CTkLabel(self.card_ounce, text="Calculating...", font=ctk.CTkFont(family="Consolas", size=24, weight="bold"), text_color="#3498db")
        self.lbl_oz_val.pack(pady=(2, 4))
        self.lbl_oz_delta = ctk.CTkLabel(self.card_ounce, text="--", font=ctk.CTkFont(size=12))
        self.lbl_oz_delta.pack(pady=(2, 12))

        # Card 2: Kilogram
        self.card_kg = ctk.CTkFrame(self.telemetry_frame, corner_radius=10, fg_color="#1e272e")
        self.card_kg.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.lbl_kg_title = ctk.CTkLabel(self.card_kg, text="KILOGRAM FORECAST", font=ctk.CTkFont(size=11, weight="bold"), text_color="#bdc3c7")
        self.lbl_kg_title.pack(pady=(12, 2))
        self.lbl_kg_val = ctk.CTkLabel(self.card_kg, text="Calculating...", font=ctk.CTkFont(family="Consolas", size=24, weight="bold"), text_color="#2ecc71")
        self.lbl_kg_val.pack(pady=(2, 4))
        self.lbl_kg_delta = ctk.CTkLabel(self.card_kg, text="--", font=ctk.CTkFont(size=12))
        self.lbl_kg_delta.pack(pady=(2, 12))

    def create_right_panel(self):
        """Creates the AI Chat Advisor panel."""
        self.right_frame = ctk.CTkFrame(self, corner_radius=15, border_width=1, border_color="#2c3e50")
        self.right_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)

        # Header Title
        chat_header = ctk.CTkLabel(
            self.right_frame, 
            text="💬 Kurdistan AI Advisor (Sulaymaniyah)", 
            font=ctk.CTkFont(family="Outfit", size=20, weight="bold")
        )
        chat_header.grid(row=0, column=0, padx=20, pady=15, sticky="w")

        # Chat History Textbox
        self.chat_history = ctk.CTkTextbox(
            self.right_frame, 
            corner_radius=10, 
            fg_color="#1e272e",
            font=ctk.CTkFont(size=14),
            border_width=1,
            border_color="#34495e"
        )
        self.chat_history.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.chat_history.insert("0.0", "Advisor Bot: سڵاو! من ڕاوێژکاری زیرەکی زیوم لە سلێمانی. چۆن دەتوانم یارمەتیت بدەم؟\n\n")
        self.chat_history.configure(state="disabled")

        # Chat Controls
        self.input_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, padx=20, pady=15, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.chat_input = ctk.CTkEntry(
            self.input_frame, 
            placeholder_text="پرسیارەکەت لێرە بنووسە بۆ ڕاوێژکاری زیو...",
            font=ctk.CTkFont(size=13),
            height=40
        )
        self.chat_input.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.chat_input.bind("<Return>", lambda event: self.send_chat_message())

        self.send_btn = ctk.CTkButton(
            self.input_frame, 
            text="بنێرە", 
            width=80, 
            height=40,
            command=self.send_chat_message,
            font=ctk.CTkFont(weight="bold")
        )
        self.send_btn.grid(row=0, column=1, sticky="e")

    # ---------------------------------------------------------------------------
    # PIPELINE FORECAST CALCULATIONS (THREADED)
    # ---------------------------------------------------------------------------

    def load_model_and_forecast(self):
        """Spins up a background thread to calculate price forecasts."""
        threading.Thread(target=self._forecast_task, daemon=True).start()

    def _forecast_task(self):
        try:
            # 1. Fetch live market indicators
            today_str = date.today().strftime("%Y-%m-%d")
            raw_history = fetch_historical_data(start="2026-02-01", end=today_str)
            cleaned_history = clean_and_engineer(raw_history)

            exclude_cols = [
                "sentiment_score", "sentiment_discrete", "headline_count",
                "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std"
            ]
            ts_cols = [col for col in cleaned_history.columns if col not in exclude_cols]

            recent_ts = cleaned_history[ts_cols].tail(20)
            self.latest_close = float(cleaned_history["close"].iloc[-1])

            # 2. Get latest sentiment
            self.sentiment_score = 0.0
            if os.path.exists("silver_daily_sentiment.csv"):
                saved_sent = pd.read_csv("silver_daily_sentiment.csv", index_col=0)
                self.sentiment_score = float(saved_sent["sentiment_score"].iloc[-1])

            # 3. Load weights & scalers
            with open("silver_scalers.pkl", "rb") as f:
                scalers = pickle.load(f)

            model = SilverPredictor(
                ts_input_dim=len(ts_cols),
                lstm_hidden_dim=64,
                lstm_layers=2,
                sentiment_dim=1,
                sentiment_hidden_dim=16,
                fc_hidden_dim=32,
                dropout=0.2
            )
            model.load_state_dict(torch.load("silver_multimodal_predictor.pth", map_location="cpu"))
            model.eval()

            # Preprocess inputs
            scaled_ts = scalers["ts"].transform(recent_ts)
            scaled_sent = scalers["sent"].transform([[self.sentiment_score]])[0][0]

            ts_tensor = torch.tensor(scaled_ts, dtype=torch.float32).unsqueeze(0)
            sent_tensor = torch.tensor([[scaled_sent]], dtype=torch.float32)

            with torch.no_grad():
                norm_pred = model(ts_tensor, sent_tensor).numpy()

            self.predicted_troy_oz = float(scalers["target"].inverse_transform(norm_pred)[0][0])
            self.predicted_kg = (self.predicted_troy_oz / 31.103) * 1000

            # Update GUI cards
            pct_change = ((self.predicted_troy_oz - self.latest_close) / self.latest_close) * 100
            sign = "+" if pct_change >= 0 else ""
            color = "#2ecc71" if pct_change >= 0 else "#e74c3c"
            arrow = "▲" if pct_change >= 0 else "▼"

            self.lbl_oz_val.configure(text=f"${self.predicted_troy_oz:.4f}")
            self.lbl_oz_delta.configure(text=f"{arrow} {sign}{pct_change:.2f}%", text_color=color)

            self.lbl_kg_val.configure(text=f"${self.predicted_kg:.2f}")
            self.lbl_kg_delta.configure(text=f"{arrow} {sign}{pct_change:.2f}%", text_color=color)

            # Embed Matplotlib Line Chart
            self.embed_chart(cleaned_history)

        except Exception as e:
            self.lbl_oz_val.configure(text="ERROR")
            self.lbl_kg_val.configure(text="ERROR")
            print(f"[Forecast Error] {e}")

    def embed_chart(self, df):
        """Draws the matplotlib validation/prediction chart directly inside the GUI."""
        fig, ax = plt.subplots(figsize=(6.5, 3.25), dpi=100)
        fig.patch.set_facecolor("#1e272e")
        ax.set_facecolor("#1e272e")

        hist_df = df["close"].tail(30)
        dates = list(hist_df.index)
        prices = list(hist_df.values)

        # Plot historical prices
        ax.plot(dates, prices, color="#3498db", label="Historical Close", linewidth=2.0)

        # Plot predicted connection point
        tomorrow = dates[-1] + timedelta(days=1)
        if tomorrow.weekday() >= 5:
            tomorrow += timedelta(days=7 - tomorrow.weekday())

        ax.plot([dates[-1], tomorrow], [prices[-1], self.predicted_troy_oz], color="#e74c3c", linestyle="--", linewidth=1.5)
        ax.scatter([tomorrow], [self.predicted_troy_oz], color="#2ecc71", s=80, marker="*", label="AI Forecast Close", zorder=5)

        ax.set_title("Silver Price Trend & AI Forecast", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        ax.spines["bottom"].set_color("#2c3e50")
        ax.spines["top"].set_color("#2c3e50")
        ax.spines["left"].set_color("#2c3e50")
        ax.spines["right"].set_color("#2c3e50")
        ax.grid(True, linestyle=":", color="#34495e", alpha=0.5)

        ax.legend(facecolor="#1e272e", edgecolor="#2c3e50", labelcolor="white", fontsize=8, loc="upper left")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.left_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

    # ---------------------------------------------------------------------------
    # AI ADVISOR CHAT ENGINE (THREADED)
    # ---------------------------------------------------------------------------

    def send_chat_message(self):
        user_text = self.chat_input.get().strip()
        if not user_text:
            return

        # Display user message in chat
        self.chat_history.configure(state="disabled")
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", f"تۆ: {user_text}\n\n")
        self.chat_history.configure(state="disabled")
        self.chat_input.delete(0, "end")

        # Disable send interface during thinking
        self.send_btn.configure(state="disabled")
        
        # Start background agent worker
        threading.Thread(target=self._advisor_task, args=(user_text,), daemon=True).start()

    def _advisor_task(self, prompt):
        try:
            # Check API keys
            groq_key = os.getenv("GROQ_API_KEY")
            tavily_key = os.getenv("TAVILY_API_KEY")

            if not groq_key or not tavily_key:
                response = "سەرچاوەی بڕوانامەکان (.env) بە دروستی دانەمەزراوە. تکایە کلیلی API بۆ Groq و Tavily دابنێ."
                self.append_to_chat(response)
                return

            # Update status in chat history
            self.append_to_chat("ڕاوێژکار: خەریکی لێکۆڵینەوەم لە نوێترین ھەواڵەکان و داتا ئابوورییەکان...", is_status=True)

            # 1. Search Tavily for recent market news
            tavily = TavilyClient(api_key=tavily_key)
            query = f"silver market news macroeconomic conditions Iraq Kurdistan Sulaymaniyah date:{date.today().year}"
            search_response = tavily.search(query=query, max_results=4)

            # Package context
            search_results = "\n\n".join([f"Title: {r['title']}\nSnippet: {r['content']}" for r in search_response['results']])

            model_telemetry = (
                f"Silver Market Telemetry:\n"
                f"- Today's Last Actual Close: ${self.latest_close:.4f} / Troy Oz\n"
                f"- Tomorrow's AI Predicted Price: ${self.predicted_troy_oz:.4f} / Troy Oz\n"
                f"- Tomorrow's Predicted Price (Kg): ${self.predicted_kg:.2f}\n"
                f"- Sentiment Score Indicator: {self.sentiment_score:+.4f}"
            )

            # 2. Invoke Groq Client
            groq = Groq(api_key=groq_key)
            system_prompt = (
                "You are an expert Silver Market Advisor located in Sulaymaniyah, Kurdistan. You speak fluent Kurdish (Sorani). "
                "Your job is to analyze the provided AI predicted prices and Tavily news data, and advise the user whether today is a good day to buy or sell silver in the local Sulaymaniyah bazaars. "
                "Be direct, professional, and base your advice purely on the data margins."
            )

            user_payload = (
                f"User Question: {prompt}\n\n"
                f"{model_telemetry}\n\n"
                f"Latest Economic & Silver News:\n{search_results}"
            )

            chat_completion = groq.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload}
                ],
                model="llama3-70b-8192",
                temperature=0.3
            )

            response = chat_completion.choices[0].message.content
            
            # Clean up status line and display answer
            self.append_to_chat(f"ڕاوێژکار: {response}")

        except Exception as e:
            self.append_to_chat(f"ڕاوێژکار: بمبەخشە، ھەڵەیەک ڕوویدا لە کاتی پەیوەندیکردن بە سێرڤەر: {e}")
        finally:
            self.send_btn.configure(state="normal")

    def append_to_chat(self, text, is_status=False):
        self.chat_history.configure(state="normal")
        if is_status:
            # Overwrite status or append cleanly
            self.chat_history.insert("end", f"{text}\n\n")
        else:
            # Remove any pending "thinking" phrases if appropriate
            self.chat_history.insert("end", f"{text}\n\n")
        self.chat_history.see("end")
        self.chat_history.configure(state="disabled")


if __name__ == "__main__":
    app = SilverDesktopApp()
    app.mainloop()
