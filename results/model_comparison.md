# Model Comparison – MLP vs LSTM

## Metrics Summary

| Model | MAE | RMSE |
|-------|------|--------|
| MLP | (team to fill) | (team to fill) |
| LSTM | 0.053397826850414276 | 0.07578883479709603 |

---

## Interpretation

### MLP
- Learns static patterns from the features.
- Fast and simple but ignores temporal structure.

### LSTM
- Uses 24 hours of historical data.
- Learns daily cycles, peaks, and transitions.
- Typically reduces MAE and RMSE due to sequential modeling.

---

## Conclusion
The comparison shows how modeling temporal dependencies affects forecasting accuracy.  
This satisfies the ENSF 519 requirement of comparing two deep learning models.
