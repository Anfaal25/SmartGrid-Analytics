"""
train_mlp.py
------------
Trains an MLP model using the SAME preprocessing and sliding windows
as the LSTM model, for fair comparison.

Outputs:
    models/mlp_model.h5
    results/mlp_metrics.json
    results/mlp_history.json
    results/mlp_predictions.csv
    results/mlp_predictions.png
    results/mlp_training_curves.png
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
# 2. Normalize data (same as LSTM)
# --------------------------------------------------------------
print("Scaling features...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# --------------------------------------------------------------
# 3. Sliding windows (same as LSTM)
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

# Flatten for MLP:
# LSTM input = (samples, 24, features)
# MLP input = (samples, 24 * features)
X = X.reshape(X.shape[0], -1)

# --------------------------------------------------------------
# 4. Train/Test split
# --------------------------------------------------------------
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# --------------------------------------------------------------
# 5. Build MLP Model
# --------------------------------------------------------------
print("Building MLP model...")

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
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
print("Training MLP...")

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# --------------------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------------------
print("Evaluating MLP...")

preds = model.predict(X_test).ravel()

mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5
mae = model.evaluate(X_test, y_test, verbose=0)[1]

metrics = {
    "RMSE": float(rmse),
    "MAE": float(mae)
}

print("MLP Metrics:", metrics)

# --------------------------------------------------------------
# 8. Save outputs
# --------------------------------------------------------------
os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

model.save("../models/mlp_model.h5")

with open("../results/mlp_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Training history
history_dict = {
    "loss": list(map(float, history.history["loss"])),
    "val_loss": list(map(float, history.history["val_loss"])),
    "mae": list(map(float, history.history["mae"])),
    "val_mae": list(map(float, history.history["val_mae"]))
}
with open("../results/mlp_history.json", "w") as f:
    json.dump(history_dict, f, indent=4)

# Predictions CSV
pd.DataFrame({
    "y_test": y_test,
    "y_pred": preds
}).to_csv("../results/mlp_predictions.csv", index=False)

# Plot 1: Predictions vs Actual
plt.figure(figsize=(12,5))
plt.plot(y_test[:300], label="Actual", linewidth=2)
plt.plot(preds[:300], label="Predicted", linewidth=2)
plt.title("MLP Predictions vs Actual (first 300 samples)")
plt.xlabel("Time")
plt.ylabel("Scaled Power")
plt.legend()
plt.tight_layout()
plt.savefig("../results/mlp_predictions.png")

# Plot 2: Training curves
plt.figure(figsize=(12,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("MLP Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("../results/mlp_training_curves.png")

print("All MLP outputs saved successfully!")
