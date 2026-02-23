# ⚡ EV Charging Demand Prediction

A Streamlit web app that predicts EV charging demand across different traffic zones using time-series regression.

## Features

- Upload CSV → select model → get instant forecasts
- **Models:** Linear Regression, Random Forest
- **Metrics:** MAE & RMSE
- **Visualizations:** Actual vs Predicted plot, Monthly demand bar chart

## Tech Stack

Python · Streamlit · scikit-learn · Pandas · Matplotlib

## Quick Start

1. **Clone & navigate**
   ```bash
   git clone https://github.com/naamyuvraj/EV-Charging-Prediction.git
   cd EV-Charging-Prediction/EV_Charging_Demand_Prediction
   ```

2. **Set up environment & install dependencies**
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
EV_Charging_Demand_Prediction/
├── app.py                # Streamlit UI
├── requirements.txt
├── data/
│   └── volume.csv        
└── src/
    ├── preprocess.py     # Timestamp parsing & feature extraction
    ├── train_model.py    # Feature engineering & model training
    └── predict.py        # Prediction & evaluation
```

## How It Works

1. **Preprocess** — Melts wide-format CSV to long format; extracts hour, day-of-week, month, weekend flag
2. **Train** — Adds cyclical encodings (sin/cos) + lag-1 feature; 80/20 chronological split
3. **Predict** — Runs inference and computes MAE & RMSE
