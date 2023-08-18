import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# Load your dataset
data = pd.read_csv('your_data.csv')

# Preprocess the data
# Assuming you have columns like 'feature_1', 'feature_2', ..., 'target'
features = ['feature_1', 'feature_2']  # Add your feature column names here
target = 'target'  # Replace with your target column name

# Scale the features and target to the range [0, 1]
scaler = MinMaxScaler()
data[features + [target]] = scaler.fit_transform(data[features + [target]])

# Split the data into train and test sets
X = data[features].values
y = data[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for GRU input (samples, timesteps, features)
timesteps = 10  # Choose the appropriate number of time steps
X_train = np.reshape(X_train, (X_train.shape[0], timesteps, len(features)))
X_test = np.reshape(X_test, (X_test.shape[0], timesteps, len(features)))

# Build the GRU model
model = Sequential()
model.add(GRU(units=50, input_shape=(timesteps, len(features))))
model.add(Dense(units=1))  # Output layer
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Make predictions for the next 7 days
prediction_days = 7
last_sequence = X_test[-1]  # Use the last sequence in the test set as a starting point

predicted_values = []
for _ in range(prediction_days):
    predicted_value = model.predict(np.array([last_sequence]))[0, 0]
    predicted_values.append(predicted_value)
    last_sequence = np.vstack((last_sequence[1:], [predicted_value]))

# Inverse transform the predicted values
predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))

# Print the predicted values
print(predicted_values)
