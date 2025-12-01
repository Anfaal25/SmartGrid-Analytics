
# ENSF 519 – SmartGrid-Analytics

## 📌 Overview
This project aims to forecast short-term household electricity consumption using deep learning models.  
We use the **Individual Household Electric Power Consumption Dataset** from the UCI Machine Learning Repository.

The goal is to compare a **baseline MLP** against a **sequence-based LSTM** and determine whether temporal modeling significantly improves forecasting accuracy.

**Proposed client:** SmartGrid Analytics  
**Use case:** Predict future electricity demand to optimize grid balancing, reduce energy waste, and anticipate peak load periods.

---


---

## 👥 Team Members & Roles
| Member | UCID | Role | Branch |
|--------|------|-------|---------|
| **Anfaal Mahbub** | 30140009 | Pair programming – LSTM Model | `feature/lstm-model` |
| **Joshua Koshy** | 30149273 | Pair programming – MLP Model | `feature/mlp-model` |
| **Mehvish Shakeel** | 30161318 | Pair programming – LSTM Model | `feature/lstm-model` |
| **Tara Cherian** | 30143816 | Pair programming – MLP Model | `feature/mlp-model` |

---


## 📊 Dataset

**Dataset:** Individual Household Electric Power Consumption
**Link:** [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

### Key Features

* Global active power (kW)
* Global reactive power (kVAR)
* Voltage (V)
* Global intensity (A)
* Sub-metering 1, 2, 3
* Timestamp (Date, Time)

### Target Variable

* **Future Global Active Power (kW)**



## 🧠 Models

### 1️⃣ **Multilayer Perceptron (MLP)**

* Baseline regression model
* Uses fully connected layers
* Predicts power consumption from current feature snapshot
* Fast and simple
* Branch: `MLP-Model`


---

### 2️⃣ **LSTM (Long Short-Term Memory)**

* Sequential model for time-series
* Learns temporal dependencies
* Uses sliding windows of past consumption
* Branch: `LSTM-Model`


---

## 📈 Evaluation Metrics

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **MSE** – Mean Squared Error

---

## ✅ Notes

* This project is part of **ENSF 519 – Deep Learning** (Fall 2025).
* Dataset or models may change for final submission if justified by results.

---
