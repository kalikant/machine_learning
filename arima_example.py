import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime, timedelta

# Generate sample data
np.random.seed(42)
date_range = pd.date_range(start="2023-01-01", periods=100, freq="D")
client_ids = np.random.randint(1, 11, size=100)
client_trade_dates = [d + timedelta(days=np.random.randint(1, 30)) for d in date_range]

data = pd.DataFrame({
    "client_id": client_ids,
    "client_trade_date": client_trade_dates
})

# Prepare data for prediction
data = data.sort_values(by="client_trade_date")
data_grouped = data.groupby("client_id")

predictions = []

for client_id, group in data_grouped:
    ts = group.set_index("client_trade_date")["client_id"]
    model = ARIMA(ts, order=(5, 1, 0))  # ARIMA(p, d, q) order
    model_fit = model.fit(disp=0)
    forecast_steps = 5
    forecast, stderr, conf_int = model_fit.forecast(steps=forecast_steps)
    
    last_date = ts.index[-1]
    for i in range(forecast_steps):
        prediction_date = last_date + timedelta(days=i+1)
        predictions.append({
            "prediction_date": prediction_date,
            "client_id": client_id,
            "client_trade_date_in_future": prediction_date + timedelta(days=int(forecast[i]))
        })

# Create predictions DataFrame
predictions_df = pd.DataFrame(predictions)

# Save predictions to a CSV file
predictions_df.to_csv("time_series_predictions.csv", index=False)
