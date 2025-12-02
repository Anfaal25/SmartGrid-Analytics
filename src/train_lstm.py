"""
train_lstm.py
-------------
Trains an LSTM model to forecast hourly electricity consumption.

Workflow:
1. Load processed dataset
2. Normalize features
3. Create 24-hour sliding windows
4. Split into train/test (time-series split)
5. Build and train LSTM model
6. Evaluate with MAE and RMSE
7. Save model, predictions, metrics, training curves

Outputs:
    models/lstm_model.h5
    results/lstm_metrics.json
    results/lstm_history.json
    results/lstm_predictions.png
    results/lstm_training_curves.png
    results/lstm_predictions.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import matplotlib.pyplot as plt
import json
import os

# --------------------------------------------------------------
# 1. Load dataset
# --------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("../data/processed_power.csv", index_col="datetime")
target_col_index = 0  # Global_active_power

# --------------------------------------------------------------
# 2. Normalize data
# --------------------------------------------------------------
print("Scaling features...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# --------------------------------------------------------------
# 3. Sliding windows
# --------------------------------------------------------------
def create_sequences(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, target_col_index])
    return np.array(X), np.array(y)

WINDOW = 24
print(f"Creating sliding windows (window={WINDOW})...")
X, y = create_sequences(scaled_data, WINDOW)

# --------------------------------------------------------------
# 4. Train/Test split
# --------------------------------------------------------------
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# --------------------------------------------------------------
# 5. Build LSTM Model
# --------------------------------------------------------------
print("Building LSTM model...")

model = keras.Sequential([
    keras.layers.Input(shape=(WINDOW, X.shape[2])),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# --------------------------------------------------------------
# 6. Train model
# --------------------------------------------------------------
print("Training LSTM...")

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --------------------------------------------------------------
# 7. Evaluate on Test Set
# --------------------------------------------------------------
print("Evaluating LSTM...")

preds = model.predict(X_test).ravel()

mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5
mae = model.evaluate(X_test, y_test, verbose=0)[1]

metrics = {
    "RMSE": float(rmse),
    "MAE": float(mae)
}

print("LSTM Metrics:", metrics)

# --------------------------------------------------------------
# 8. Save model + metrics + history
# --------------------------------------------------------------
os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

model.save("../models/lstm_model.h5")

with open("../results/lstm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save training history
history_dict = {
    "loss": list(map(float, history.history["loss"])),
    "val_loss": list(map(float, history.history["val_loss"])),
    "mae": list(map(float, history.history["mae"])),
    "val_mae": list(map(float, history.history["val_mae"]))
}
with open("../results/lstm_history.json", "w") as f:
    json.dump(history_dict, f, indent=4)

# Save raw predictions
pred_df = pd.DataFrame({
    "y_test": y_test,
    "y_pred": preds
})
pred_df.to_csv("../results/lstm_predictions.csv", index=False)

# --------------------------------------------------------------
# 9. Plot: Predictions vs Actual
# --------------------------------------------------------------
print("Saving prediction plot...")
plt.figure(figsize=(12,5))
plt.plot(y_test[:300], label="Actual", linewidth=2)
plt.plot(preds[:300], label="Predicted", linewidth=2)
plt.title("LSTM Predictions vs Actual (first 300 samples)")
plt.xlabel("Time (hours)")
plt.ylabel("Scaled Power")
plt.legend()
plt.tight_layout()
plt.savefig("../results/lstm_predictions.png")

# --------------------------------------------------------------
# 10. Plot: Training Curves
# --------------------------------------------------------------
print("Saving training curves...")
plt.figure(figsize=(12,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("../results/lstm_training_curves.png")

plt.figure(figsize=(12,5))
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("Training & Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.savefig("../results/lstm_mae_curves.png")

print("All outputs saved successfully!")
