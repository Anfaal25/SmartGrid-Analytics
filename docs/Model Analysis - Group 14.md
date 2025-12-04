# **Model Comparison - SmartGrid Analytics (ENSF 519)**

### **MLP vs LSTM for Hourly Electricity Consumption Forecasting**

This file summarizes the performance of the two deep learning models trained in the project:

* **MLP (Multilayer Perceptron)** - Baseline
* **LSTM (Long Short-Term Memory)** - Sequence Model

Both models use **the same preprocessing**, **sliding windows**, **train/test split**, and **evaluation metrics** to ensure a fair comparison.

---

# **Final Performance Metrics**

| Model    | MAE        | RMSE       | Notes                                               |
| -------- | ---------- | ---------- | --------------------------------------------------- |
| **MLP**  | **0.0537** | **0.0789** | Strong baseline but cannot model temporal structure |
| **LSTM** | **0.0516** | **0.0745** | Best performance - learns time dependencies         |

**Lower = better**

---

# **Interpretation of Metrics**

### **1. LSTM achieves lower MAE and RMSE**

* The LSTM’s **MAE is ~4% lower** than the MLP
* The LSTM’s **RMSE is ~6% lower**
* This means:

  * Smaller average error
  * Fewer large mistakes
  * Smoother, more accurate predictions

### **2. Why LSTM outperforms MLP**

* Electricity consumption is **highly time-dependent**
  (daily cycles, appliance usage patterns, weekday/weekend differences)
* LSTM processes **24-hour sequences**, learning:

  * Routines
  * Short-term fluctuations
  * Morning/evening peaks
* The MLP treats the past 24 hours as a **flat vector with no sequence order**, so it cannot capture temporal structure.

**Result: LSTM generalizes better on unseen data.**

---

# **Training Behavior Analysis**

### **LSTM Training Pattern**

* Starts with `mae ≈ 0.0770`
* Gradually decreases to `mae ≈ 0.0564`
* Validation MAE consistently improves down to **0.0516**
* No sign of instability or exploding gradients
* Converges smoothly over 20 epochs

**Stable training, good generalization.**

---

### **MLP Training Pattern**

* Starts with `mae ≈ 0.0705`
* Reaches training MAE of **0.0485** - the model fits the training data well
* But validation MAE fluctuates between `0.051–0.058`
* Some epochs show rising validation loss = **mild overfitting**

MLP fits the trend but cannot fully capture sequential relationships.

---

# **Qualitative Prediction Comparison**

### **LSTM Predictions**

* Tracks peaks and valleys more accurately
* Better at capturing abrupt changes
* Less noisy, smoother fit

### **MLP Predictions**

* Follows general trend
* Lagging response during sudden increases/decreases
* Slightly more variance around true signal

Overall, the LSTM is **consistently closer** to actual power usage.

---

# **Conclusion**

Based on both quantitative metrics and qualitative behavior:

**The LSTM model provides the best overall accuracy, stability, and trend-following capability for forecasting hourly electricity consumption.**

This confirms the project hypothesis:

Models that learn temporal structure (LSTM)
outperform models that treat inputs as static vectors (MLP)
**on time-series prediction tasks like electricity consumption forecasting.**

---


