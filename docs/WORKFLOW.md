# Machine Learning Workflow – SmartGrid Analytics

This project follows the standard ML workflow:

---

## 1. Data Acquisition
Dataset is downloaded from UCI ML Repository.

Files:
- `household_power_consumption.txt`
- Converted to CSV for efficiency.

---

## 2. Data Preprocessing
Performed in `preprocess.py`:
- Parse datetime from Date + Time
- Convert numeric columns
- Clean missing values
- Resample to hourly averages
- Save processed dataset

Output:
- `processed_power.csv`

---

## 3. Feature Engineering
- Sliding windows: 24 hours used to predict the next hour.
- Normalization using MinMaxScaler.

---

## 4. Model Development

### MLP
- Baseline dense network for comparison.

### LSTM
- Sequential model capturing temporal dependencies.
- Architecture:
  - LSTM(64) → LSTM(32) → Dense(1)

---

## 5. Training and Validation
- 80/20 time-based split.
- Loss: MSE  
- Metrics: MAE, RMSE  

---

## 6. Evaluation
Metrics saved as:
- `results/lstm_metrics.json`


Visualization:
- `results/lstm_predictions.png`
- `results/lstm_training_curves.png`
- `results/lstm_mae_curves.png`


---

## 7. Comparison Report
See: `results/model_comparison.md`

---

## 8. Deployment (Future Work)
- Could be integrated into a SmartGrid monitoring dashboard.
