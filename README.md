# SmartGrid Analytics – Electricity Consumption Forecasting  
ENSF 519 Final Project (Group 14)

This project forecasts **hourly household electricity usage** using two deep learning models:

- **MLP (Multilayer Perceptron)** – baseline model  
- **LSTM (Long Short-Term Memory)** – sequence model  

Both models use the **UCI Household Power Consumption Dataset**.

---

# 📁 Repository Structure

```

SmartGrid-Analytics/
│
├── data/
│   ├── household_power_consumption.txt        # Raw UCI dataset
│   ├── household_power_consumption.csv        # Converted CSV (Step 1)
│   └── processed_power.csv                    # Cleaned hourly dataset (Step 2)
│
├── src/
│   ├── convert_txt_to_csv.py                  # Convert .txt → .csv
│   ├── preprocess.py                          # Clean + resample data
│   ├── train_mlp.py                           # Baseline model (dense network)
│   └── train_lstm.py                          # Sequence model (LSTM)
│   
│
├── models/
│   ├── mlp_model.h5                           # Saved MLP
│   └── lstm_model.h5                          # Saved LSTM
│
├── results/
│   ├── lstm_metrics.json                      # LSTM MAE + RMSE
│   ├── lstm_history.json                      # LSTM training curves (loss/MAE)
│   ├── lstm_predictions.csv                   # Raw predictions vs actual
│   ├── lstm_predictions.png                   # Plot: actual vs predicted
│   ├── lstm_training_curves.png               # Training vs validation loss
│   ├── lstm_mae_curves.png                    # Training vs validation MAE
│   │
│   ├── mlp_metrics.json                       # MLP MAE + RMSE
│   ├── mlp_history.json                       # MLP training curves
│   ├── mlp_predictions.csv                    # Raw predictions vs actual
│   ├── mlp_predictions.png                    # Plot: actual vs predicted
│   └── mlp_training_curves.png                # Training curves
│   
│   
│
├── docs/
│   ├── TEAM_ROLES.md                          # Pair programming + roles
│   └── model_comparison.md                    # Final table comparing models
│
└── README.md                                  # Main project instructions



````

---

#  **How to Run the Project**

Below are the **three required steps** to go from raw data → final models.

---

#  **Step 1 — Convert `.txt` → `.csv`**  
(UCI dataset is originally semi-colon separated with `?` as missing values.)

Run:

```bash
python src/convert_txt_to_csv.py
````

This produces:

```
data/household_power_consumption.csv
```

This CSV contains **minute-level** readings exactly as provided by UCI.

---

#  **Step 2 — Preprocess the dataset**

The LSTM and MLP cannot train on raw UCI data.
So we clean, convert types, and resample it to **hourly averages**.

Run:

```bash
python src/preprocess.py
```

This script:

✔ Combines Date + Time into a single timestamp

✔ Converts all numeric fields

✔ Handles missing values

✔ Resamples to **hourly data**

✔ Drops invalid rows

✔ Saves processed dataset


Output:

```
data/processed_power.csv
```

This file is used by **both models**.

---

#  **Step 3 — Train the Models**

#  Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---
# Notes

* All scripts run entirely on CPU (GPU optional).
* Dataset is included inside the `/data` folder for reproducibility.
* This workflow satisfies the ENSF-519 project rubric:

  * Proper ML workflow
  * Data preprocessing
  * Comparison of two deep learning models
  * Documentation + reproducibility
  * Clean directory organization
  * Evaluation metrics and plots

---

Once `processed_power.csv` exists, you can train:



# **3A — Run the MLP Model (Baseline)**

```bash
python src/train_mlp.py
```

This script:

* Loads `processed_power.csv`
* Normalizes features
* Trains a dense neural network
* Evaluates using MAE and RMSE
* Saves model + metrics

Outputs:

```
models/mlp_model.h5
results/mlp_metrics.json
results/mlp_history.json
results/mlp_predictions.csv
results/mlp_predictions.png
results/mlp_training_curves.png
```





# **3B — Run the LSTM Model (Sequence Model)**

```bash
python src/train_lstm.py
```

This script:

* Loads `processed_power.csv`
* Creates 24-hour sliding windows
* Builds a 2-layer LSTM
* Trains for 20 epochs
* Evaluates using MAE and RMSE
* Saves training history
* Saves plots and predictions

Outputs:

```
models/lstm_model.h5
results/lstm_metrics.json
results/lstm_history.json
results/lstm_predictions.csv
results/lstm_predictions.png
results/lstm_training_curves.png
results/lstm_mae_curves.png
```





# **Interpretation of Metrics**

Both models report two standard regression metrics:

### • MAE (Mean Absolute Error)

Measures the *average* prediction error.
Lower MAE = better average accuracy.

### • RMSE (Root Mean Squared Error)

Penalizes **larger errors** more heavily.
Lower RMSE = fewer large mistakes.



### Actual Model Performance

```
MLP  →  MAE = 0.0596   |   RMSE = 0.0857
LSTM →  MAE = 0.0534   |   RMSE = 0.0758
```

### LSTM is more accurate

Both MAE and RMSE are lower for the LSTM:

* **LSTM MAE is ~10% lower** than the MLP
* **LSTM RMSE is ~12% lower** than the MLP

This shows that the LSTM makes:

* **smaller average errors**, and
* **fewer large mistakes**, compared to the MLP.

---

### Interpretation

* The **MLP** learns basic input-output relationships but has **no understanding of temporal patterns** in electricity usage.
* The **LSTM**, using a 24-hour sliding window, learns daily cycles (morning/evening peaks), appliance usage rhythms, and short-term trends.
* Because electricity consumption is highly time-dependent, the LSTM’s sequential structure gives it a **natural advantage**.

---

### Conclusion

The LSTM model **outperforms the MLP** on both evaluation metrics, demonstrating that:

**Modeling the temporal structure in power consumption significantly improves forecasting accuracy.**

This aligns with the project goal and validates the decision to compare a baseline (MLP) with a sequence model (LSTM).



---
# Team Roles – SmartGrid Analytics (ENSF 519)

## Group Members
- **Anfaal Mahbub (30140009)** – LSTM Model Co-Developer
- **Joshua Koshy (30149273)** – MLP Model Co-Developer
- **Mehvish Shakeel (30161318)** – LSTM Pair Programming Partner
- **Tara Cherian (30143816)** – MLP Pair Programming Partner

---

## Pair Programming Structure
We use two pairing teams:

### Pair 1 — LSTM Team
- **Anfaal + Mehvish**
- Responsibilities:
  - Sequence creation
  - LSTM architecture + training
  - LSTM evaluation metrics
  - Prediction plotting
  - Saving model + metrics

### Pair 2 — MLP Team
- **Joshua + Tara**
- Responsibilities:
  - Build baseline MLP model
  - Train + evaluate MLP
  - Ensure outputs match the LSTM format
  - Contribute to comparison analysis

---

## Shared Responsibilities
- Writing final documentation
- Maintaining clear GitHub commits
- Reviewing model comparison
- Ensuring smooth workflow across branches
  
---

## Citations

G. Hebrail and A. Berard. "Individual Household Electric Power Consumption," UCI Machine Learning Repository, 2006. [Online]. Available: https://doi.org/10.24432/C58K54.

Generative AI was used for debugging and syntax.
OpenAI. (2025). ChatGPT (Dec 2 version) [Large language model]. https://chat.openai.com/chat
 
