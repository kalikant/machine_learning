import numpy as np
from numpy import array
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import json
from pandas.io.json import json_normalize

N_STEPS = 3
N_FEATURES = 1


def read_data_from_file():
    file_name = '/Users/kalikantjha/PycharmProjects/TradeIdea/venv/data/AAPL_1980-12-12_to_2019-06-17.json'
    with open(file_name) as file:
        historical_stock_prices = json.load(file)
    df_prices = json_normalize(historical_stock_prices['AAPL']['prices'])
    df_prices = df_prices[['formatted_date', 'close']]
    df_prices = df_prices.rename(columns={'formatted_date': 'date'})
    df_prices.index = df_prices.date
    #df_prices = df_prices.sort_index

    return df_prices


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def train_model():
    data = read_data_from_file()
    raw_seq = np.array(data.close.values).ravel().tolist()
    X, y = split_sequence(raw_seq, N_STEPS)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], N_FEATURES))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(N_STEPS, N_FEATURES)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    model.save('/Users/kalikantjha/PycharmProjects/TradeIdea/venv/models/AAPL_1980-12-12_to_2019-06-17_V_LSTM.h5')
    print('LSTM_V model has trained succesfully .. ')


def prediction_accuracy_check(test_data, prediction_data):
    return np.sqrt(np.mean(np.power((test_data - prediction_data), 2)))


def prediction(input_data):
    x_input = input_data
    n_steps = 3
    n_features = 1
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(yhat)


def main():
    #train_model()
    model = load_model(
        '/Users/kalikantjha/PycharmProjects/TradeIdea/venv/models/AAPL_1980-12-12_to_2019-06-17_V_LSTM.h5')

    data = read_data_from_file()
    raw_seq = np.array(data.close.values).ravel().tolist()
    # predicting next 30 days based on last 365 days
    for i in range(30):
        raw_seq = raw_seq[(-365 + i):]
        X, y = split_sequence(raw_seq, N_STEPS)
        X = X.reshape((X.shape[0], X.shape[1], N_FEATURES))
        closing_price = model.predict(X, verbose=1)
        print(closing_price)
        #del raw_seq[i]
        #raw_seq.append(closing_price)
        #print('Day : {} -> Predicted Closing Price : {} '.format(i, closing_price))


if __name__ == "__main__":
    main()
