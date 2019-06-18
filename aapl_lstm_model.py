from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
from pandas.io.json import json_normalize
import pandas as pd
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
import json
import csv
import pickle


file_name = '/Users/kalikantjha/PycharmProjects/TradeIdea/venv/data/AAPL_1980-12-12_to_2019-06-17.json'

historical_stock_prices = None
with open(file_name) as file:
    historical_stock_prices = json.load(file)

df_prices = json_normalize(historical_stock_prices['AAPL']['prices'],
                           meta=['formatted_date','high','low','open','close','adjclose','volume'])
#print(df_prices)


#creating dataframe
data = df_prices.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df_prices)),columns=['formatted_date', 'close'])
for i in range(0,len(data)):
    new_data['formatted_date'][i] = data['formatted_date'][i]
    new_data['close'][i] = data['close'][i]

#setting index
new_data.index = new_data.formatted_date
new_data.drop('formatted_date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:100,:]
valid = dataset[100:200,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

print(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms