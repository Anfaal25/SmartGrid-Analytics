# Design Notes – SmartGrid Analytics

## Why MLP + LSTM?
MLP:
- Provides a fast, simple baseline.
- Tests whether static input is enough.

LSTM:
- Captures temporal structure in electricity consumption.
- Learns trends such as daily cycles, peaks, and seasonal effects.

---

## Window Size Choice: 24 Hours
Electricity usage strongly depends on day–night cycles.
A 24-hour window captures:
- Morning peaks
- Evening peaks
- Sleep hours
- Daily appliance usage patterns

---

## Why MinMaxScaler?
- Ensures stable training for LSTMs
- Prevents gradient issues

---

## Why 80/20 Split (Time-Based)?
Shuffling breaks time relationships.
We respect chronological order.

---

## Directory Layout Rationale
- `/data` separates datasets
- `/src` stores pure Python scripts
- `/models` stores model weights
- `/results` keeps evaluation artifacts
- `/docs` holds documentation 
