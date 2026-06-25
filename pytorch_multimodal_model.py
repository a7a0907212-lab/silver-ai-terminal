"""
=============================================================================
 PYTORCH MULTI-MODAL MODEL FOR SILVER FUTURE PRICE FORECASTING
=============================================================================
 Author  : Auto-generated for Advanced AI Time-Series Forecasting
 Model   : SilverPredictor (nn.Module)
           - LSTM Branch: Processes daily time-series features
           - Linear Branch: Processes daily news sentiment scores
           - Concatenation & FC Head: Regularized with Dropout
 Task    : Predicts the next day's 'Close' price
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# MODULE 1 — DATA MERGING & ALIGNMENT
# ---------------------------------------------------------------------------

def merge_silver_and_sentiment(
    silver_csv_path: str = "silver_ML_dataset.csv",
    sentiment_csv_path: str = "silver_daily_sentiment.csv",
    output_csv_path: str = "silver_sentiment_ML_dataset.csv"
) -> pd.DataFrame:
    """
    Merges historical Silver features with daily news sentiment scores on the date index.
    Fills missing sentiment dates with a neutral score of 0.0.

    Parameters
    ----------
    silver_csv_path    : str - Path to silver features CSV
    sentiment_csv_path : str - Path to daily sentiment scores CSV
    output_csv_path    : str - Optional path to save merged output CSV

    Returns
    -------
    pd.DataFrame - Merged and cleaned DataFrame
    """
    print(f"\n{'='*62}")
    print(f"  STEP 1 — MERGING SILVER DATA & NEWS SENTIMENT")
    print(f"{'='*62}")

    if not os.path.exists(silver_csv_path):
        raise FileNotFoundError(f"[ERROR] Silver features dataset not found at: {silver_csv_path}")

    # Load silver features
    silver_df = pd.read_csv(silver_csv_path, parse_dates=True, index_col=0)
    silver_df.index = pd.to_datetime(silver_df.index).normalize()
    print(f"  [Silver] Loaded: {silver_df.shape[0]:,} rows x {silver_df.shape[1]} columns")

    # Load sentiment features (handle absence gracefully for testing/robustness)
    sentiment_cols = ["sentiment_score", "sentiment_discrete", "headline_count",
                      "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std"]
    
    if os.path.exists(sentiment_csv_path):
        sentiment_df = pd.read_csv(sentiment_csv_path, parse_dates=True, index_col=0)
        sentiment_df.index = pd.to_datetime(sentiment_df.index).normalize()
        print(f"  [Sentiment] Loaded: {sentiment_df.shape[0]:,} rows x {sentiment_df.shape[1]} columns")
        
        # Merge on Date index
        merged_df = silver_df.join(sentiment_df[sentiment_cols], how="left")
    else:
        print(f"  [WARN] Sentiment file '{sentiment_csv_path}' not found.")
        print(f"         Generating neutral sentiment (0.0) placeholder columns.")
        merged_df = silver_df.copy()
        for col in sentiment_cols:
            if col == "headline_count":
                merged_df[col] = 0
            else:
                merged_df[col] = 0.0

    # Fill NaN values with neutral values
    fill_vals = {
        "sentiment_score": 0.0,
        "sentiment_discrete": 0.0,
        "headline_count": 0,
        "positive_ratio": 0.0,
        "negative_ratio": 0.0,
        "neutral_ratio": 1.0,  # 100% neutral if no news
        "sentiment_std": 0.0
    }
    
    for col, val in fill_vals.items():
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(val)

    # Convert headline count to int
    if "headline_count" in merged_df.columns:
        merged_df["headline_count"] = merged_df["headline_count"].astype(int)

    # Sort index chronologically
    merged_df.sort_index(inplace=True)

    if output_csv_path:
        merged_df.to_csv(output_csv_path)
        print(f"  [OK] Merged dataset saved to: '{output_csv_path}'")
        print(f"       Merged shape: {merged_df.shape[0]:,} rows x {merged_df.shape[1]} columns")

    return merged_df


# ---------------------------------------------------------------------------
# MODULE 2 — PYTORCH DATASET & SEQUENCING
# ---------------------------------------------------------------------------

class SilverMultimodalDataset(Dataset):
    """
    Custom Dataset for Multi-modal Silver Price forecasting.
    Returns:
        - X_ts  : Time-series features history tensor (seq_len, num_ts_features)
        - X_sent: Sentiment score/features of the last step tensor (sentiment_dim,)
        - y     : Target next-day close price tensor (1,)
    """
    def __init__(self, X_ts, X_sent, y):
        self.X_ts = torch.tensor(X_ts, dtype=torch.float32)
        self.X_sent = torch.tensor(X_sent, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_ts[idx], self.X_sent[idx], self.y[idx]


def prepare_data_loaders(
    df: pd.DataFrame,
    seq_len: int = 30,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    ts_features: list = None,
    sentiment_feature: str = "sentiment_score",
    target_feature: str = "close"
):
    """
    Preprocesses the merged DataFrame:
      - Splits into train and validation sets chronologically (prevent data leakage).
      - Standardizes/Scales features and target independently.
      - Creates sliding-window sequences for LSTM.
      - Bundles sequences into PyTorch DataLoaders.
    """
    # Ensure no NaN values exist in the features or targets by cleaning
    df = df.copy()
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Define default time-series feature list if none provided
    if ts_features is None:
        exclude_cols = [
            "sentiment_score", "sentiment_discrete", "headline_count",
            "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std"
        ]
        ts_features = [col for col in df.columns if col not in exclude_cols]

    print(f"\n[Data] Preparing datasets:")
    print(f"       Time-series features: {len(ts_features)} columns")
    print(f"       Sentiment feature   : {sentiment_feature}")
    print(f"       Target feature      : {target_feature}")
    print(f"       Sequence length     : {seq_len} days")

    # Chronological Split index
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    val_df   = df.iloc[split_idx:]

    print(f"       Train set size      : {len(train_df):,} rows")
    print(f"       Val set size        : {len(val_df):,} rows")

    # Scalers
    scaler_ts = StandardScaler()
    scaler_sent = MinMaxScaler(feature_range=(-1, 1)) # Keep sentiment between -1 and 1
    scaler_target = StandardScaler()

    # Fit scalers on TRAIN data only to prevent look-ahead bias/leakage
    train_ts_scaled = scaler_ts.fit_transform(train_df[ts_features])
    train_sent_scaled = scaler_sent.fit_transform(train_df[[sentiment_feature]])
    train_target_scaled = scaler_target.fit_transform(train_df[[target_feature]])

    # Transform VAL data
    val_ts_scaled = scaler_ts.transform(val_df[ts_features])
    val_sent_scaled = scaler_sent.transform(val_df[[sentiment_feature]])
    val_target_scaled = scaler_target.transform(val_df[[target_feature]])

    # Helper function to generate sequences
    def _create_xy_sequences(ts_data, sent_data, target_data):
        X_ts_seq, X_sent_seq, y_seq = [], [], []
        # Since target is for the next day, target_data[i] is the target for sequence ending at i-1
        for i in range(seq_len, len(ts_data)):
            X_ts_seq.append(ts_data[i - seq_len : i])
            X_sent_seq.append(sent_data[i - 1])  # Sentiment of the latest day in sequence
            y_seq.append(target_data[i])         # Target close price on day i
        return np.array(X_ts_seq), np.array(X_sent_seq), np.array(y_seq)

    X_train_ts, X_train_sent, y_train = _create_xy_sequences(train_ts_scaled, train_sent_scaled, train_target_scaled)
    X_val_ts, X_val_sent, y_val = _create_xy_sequences(val_ts_scaled, val_sent_scaled, val_target_scaled)

    # Create PyTorch datasets
    train_dataset = SilverMultimodalDataset(X_train_ts, X_train_sent, y_train)
    val_dataset   = SilverMultimodalDataset(X_val_ts, X_val_sent, y_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"       Sequenced Train size: {len(train_dataset):,} samples")
    print(f"       Sequenced Val size  : {len(val_dataset):,} samples")

    scalers = {
        "ts": scaler_ts,
        "sent": scaler_sent,
        "target": scaler_target
    }
    
    return train_loader, val_loader, scalers, val_df, target_feature


# ---------------------------------------------------------------------------
# MODULE 3 — MULTI-MODAL NEURAL NETWORK
# ---------------------------------------------------------------------------

class SilverPredictor(nn.Module):
    """
    Multi-modal PyTorch neural network for Silver Futures forecasting.
    - LSTM Branch: models temporal dependencies of numerical indicators.
    - Sentiment Branch: extracts representations from news sentiment score.
    - Combined Heads: concatenates outputs and maps to prediction.
    """
    def __init__(
        self,
        ts_input_dim: int,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 2,
        sentiment_dim: int = 1,
        sentiment_hidden_dim: int = 16,
        fc_hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        super(SilverPredictor, self).__init__()

        # 1. LSTM Branch
        self.lstm = nn.LSTM(
            input_size=ts_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # 2. Linear Sentiment Branch
        self.sentiment_fc = nn.Sequential(
            nn.Linear(sentiment_dim, sentiment_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 3. Concatenation and Dense Forecasting Heads
        combined_dim = lstm_hidden_dim + sentiment_hidden_dim
        self.fc_head = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, 1)  # Predicts 1 continuous value (normalized close price)
        )

    def forward(self, x_ts, x_sent):
        # x_ts:   (batch_size, seq_len, ts_input_dim)
        # x_sent: (batch_size, sentiment_dim)

        # Process Time-Series features via LSTM
        lstm_out, _ = self.lstm(x_ts)  # (batch_size, seq_len, lstm_hidden_dim)
        lstm_last   = lstm_out[:, -1, :]  # Extract last sequence state -> (batch_size, lstm_hidden_dim)

        # Process Sentiment features via Linear layer
        sent_out = self.sentiment_fc(x_sent)  # (batch_size, sentiment_hidden_dim)

        # Concatenate multi-modal branches
        combined = torch.cat((lstm_last, sent_out), dim=1)  # (batch_size, lstm_hidden_dim + sentiment_hidden_dim)

        # Map to final prediction
        prediction = self.fc_head(combined)  # (batch_size, 1)
        return prediction


# ---------------------------------------------------------------------------
# MODULE 4 — TRAINING & VALIDATION LOOPS
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    patience: int = 10
):
    """
    Trains the SilverPredictor network using early stopping.
    """
    print(f"\n[Train] Starting model training on: {device}")
    model = model.to(device)
    
    # We use HuberLoss (Smooth L1) as it is robust to financial outliers/shocks
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(1, epochs + 1):
        # ── Training step ──────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_ts, batch_sent, batch_y in train_loader:
            batch_ts, batch_sent, batch_y = batch_ts.to(device), batch_sent.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_ts, batch_sent)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to stabilize training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * batch_ts.size(0)

        train_loss /= len(train_loader.dataset)

        # ── Validation step ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_ts, batch_sent, batch_y in val_loader:
                batch_ts, batch_sent, batch_y = batch_ts.to(device), batch_sent.to(device), batch_y.to(device)
                outputs = model(batch_ts, batch_sent)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_ts.size(0)

        val_loss /= len(val_loader.dataset)
        
        # Schedule LR based on val loss
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Print stats
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"        Epoch {epoch:>2}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"        [Early Stop] Triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"[Train] Model training finished. Best Val Loss: {best_val_loss:.6f}\n")
    return model, history


# ---------------------------------------------------------------------------
# MODULE 5 — EVALUATION & PLOTTING
# ---------------------------------------------------------------------------

def evaluate_model(model, val_loader, scalers, val_df, seq_len, target_col, device):
    """
    Evaluates the model on validation data, inverse-transforms prediction scores,
    and returns MSE, MAE metrics along with predicted vs actual values.
    """
    model.eval()
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

    # Inverse transform to original price values
    target_scaler = scalers["target"]
    inv_preds = target_scaler.inverse_transform(all_preds)
    inv_actuals = target_scaler.inverse_transform(all_actuals)

    # Calculate metrics
    mse = np.mean((inv_preds - inv_actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(inv_preds - inv_actuals))

    print(f"{'='*62}")
    print(f"  VALIDATION EVALUATION METRICS (Original Prices)")
    print(f"{'='*62}")
    print(f"  Mean Squared Error (MSE)      : {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE)     : {mae:.4f}")
    print(f"{'='*62}\n")

    # Align dates for plotting
    plot_dates = val_df.index[seq_len:]
    eval_df = pd.DataFrame({
        "Actual": inv_actuals.flatten(),
        "Predicted": inv_preds.flatten()
    }, index=plot_dates)

    return eval_df, {"mse": mse, "rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ---------------------------------------------------------------------------

def run_multimodal_pipeline(
    silver_csv: str = "silver_ML_dataset.csv",
    sentiment_csv: str = "silver_daily_sentiment.csv",
    merged_csv: str = "silver_sentiment_ML_dataset.csv",
    seq_len: int = 20,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.001
):
    """
    Runs the complete model setup, data load, train, and evaluation pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Merge datasets
    merged_df = merge_silver_and_sentiment(
        silver_csv_path=silver_csv,
        sentiment_csv_path=sentiment_csv,
        output_csv_path=merged_csv
    )

    # 2. Get features
    exclude_cols = [
        "sentiment_score", "sentiment_discrete", "headline_count",
        "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std"
    ]
    ts_cols = [col for col in merged_df.columns if col not in exclude_cols]

    # 3. Create loaders
    train_loader, val_loader, scalers, val_df, target_col = prepare_data_loaders(
        df=merged_df,
        seq_len=seq_len,
        batch_size=batch_size,
        ts_features=ts_cols,
        sentiment_feature="sentiment_score",
        target_feature="close"
    )

    # 4. Instantiate model
    model = SilverPredictor(
        ts_input_dim=len(ts_cols),
        lstm_hidden_dim=64,
        lstm_layers=2,
        sentiment_dim=1,
        sentiment_hidden_dim=16,
        fc_hidden_dim=32,
        dropout=0.2
    )

    # 5. Train
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        patience=10
    )

    # 6. Evaluate
    eval_df, metrics = evaluate_model(
        model=model,
        val_loader=val_loader,
        scalers=scalers,
        val_df=val_df,
        seq_len=seq_len,
        target_col=target_col,
        device=device
    )

    # 7. Save model and scalers
    model_path = "silver_multimodal_predictor.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  [SAVE] Saved PyTorch model weights to: '{model_path}'")
    
    import pickle
    scalers_path = "silver_scalers.pkl"
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"  [SAVE] Saved fitted scalers to: '{scalers_path}'")
    print(f"  [SAVE] Saved PyTorch model weight weights to: '{model_path}'")

    return model, eval_df, metrics, history


if __name__ == "__main__":
    run_multimodal_pipeline()
