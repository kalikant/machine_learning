import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Load data from CSV
data = pd.read_csv('your_time_series_data.csv')

# Convert trade date to datetime
data['client_trade_date'] = pd.to_datetime(data['client_trade_date'])

# Sort data by trade date
data.sort_values(by='client_trade_date', inplace=True)

# Create a pivot table with trade dates as rows and client_ids as columns
pivot_data = data.pivot(index='client_trade_date', columns='client_id', values='your_target_column')

# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(pivot_data.values)

# Prepare data for LSTM
sequence_length = 10  # You can adjust this based on your data and problem
X, y = [], []
for i in range(len(normalized_data) - sequence_length):
    X.append(normalized_data[i:i+sequence_length])
    y.append(normalized_data[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(y_train.shape[1]))  # Adjust the output size based on your data

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=2)

# Make predictions
predictions = model.predict(X_test)

# Convert predictions back to original scale
predictions_original_scale = scaler.inverse_transform(predictions)

# Create a DataFrame of predictions with dates
prediction_dates = [data.index[train_size + i + sequence_length] for i in range(len(predictions))]
predictions_df = pd.DataFrame(predictions_original_scale, index=prediction_dates, columns=pivot_data.columns)

# Save predictions to a CSV file
predictions_df.to_csv("lstm_predictions.csv")
