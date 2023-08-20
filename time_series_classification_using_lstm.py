import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, concatenate
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load your time series data from CSV
data = pd.read_csv('your_time_series_data.csv')

# Convert client IDs to categorical values
data['client_id'] = data['client_id'].astype('category')
num_clients = len(data['client_id'].cat.categories)

# Drop unnecessary columns and convert timestamp to datetime
data = data.drop(columns=['timestamp'])  # Drop if not needed
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Convert data to numpy array
data_array = data.values

# Normalize numerical features
scaler = MinMaxScaler()
data_array[:, 2:] = scaler.fit_transform(data_array[:, 2:])  # Assuming client_id is column 1

# Define the number of time steps to consider for each sequence
sequence_length = 10  # Adjust as needed

# Create sequences of data for training
X, y, client_ids = [], [], []
for i in range(len(data_array) - sequence_length):
    X.append(data_array[i:i+sequence_length, 1:])  # Client ID excluded
    y.append(data_array[i+sequence_length, 1])  # Target value to predict
    client_ids.append(data_array[i+sequence_length, 0])  # Client ID

X = np.array(X)
y = np.array(y)
client_ids = np.array(client_ids)

# Split data into train and test sets
X_train, X_test, y_train, y_test, client_ids_train, client_ids_test = train_test_split(
    X, y, client_ids, test_size=0.2, random_state=42
)

# Create input layers for client IDs and features
client_id_input = Input(shape=(1,))
features_input = Input(shape=(X_train.shape[1], X_train.shape[2]))

# Embedding layer for client IDs
client_id_embedded = Embedding(input_dim=num_clients, output_dim=10)(client_id_input)

# Concatenate the client ID embedding with the features
concatenated = concatenate([client_id_embedded, features_input])

# Build the LSTM model
lstm_out = LSTM(50)(concatenated)
output = Dense(1)(lstm_out)

model = Model(inputs=[client_id_input, features_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Set up early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    [client_ids_train, X_train], y_train,
    epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss = model.evaluate([client_ids_test, X_test], y_test)
print(f'Test Loss: {test_loss:.4f}')
