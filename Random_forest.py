import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta

# Load your dataset
data = pd.read_csv('your_data.csv')

# Preprocess the data
data['prediction_date'] = pd.to_datetime(data['prediction_date'])
data['trading_date'] = pd.to_datetime(data['trading_date'])
data['days_before'] = (data['prediction_date'] - data['trading_date']).dt.days

# Convert categorical data to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['trading_sector'])

# Prepare the features and target
X = data[['days_before', 'other_features']]
y = data['will_book_trade']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions for the next 7 days
prediction_date = pd.to_datetime('2023-08-18')
days_to_predict = 7
prediction_end_date = prediction_date + timedelta(days=days_to_predict)

prediction_data = pd.DataFrame({
    'days_before': [(prediction_end_date - pd.to_datetime('trading_date')).days],
    'other_features': [your_other_features_here]
})

predicted_probs = classifier.predict_proba(prediction_data)[:, 1]

# Filter predictions above a certain threshold
prediction_threshold = 0.5
predicted_sectors = data.columns[7:]  # Assuming columns from 8 onwards are the sector columns

sectors_to_trade = []
for i, prob in enumerate(predicted_probs):
    if prob > prediction_threshold:
        sector = predicted_sectors[i]
        sectors_to_trade.append(sector)

# Create the output dataframe
output_df = pd.DataFrame({
    'prediction_date': [prediction_end_date] * len(sectors_to_trade),
    'client_id': [your_client_id] * len(sectors_to_trade),
    'trading_sector': sectors_to_trade
})

# Save the output to a CSV file
output_df.to_csv('output_predictions.csv', index=False)
