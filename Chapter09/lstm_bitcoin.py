# Deep Learning Quick Reference Chapter 9: RNNs for time series prediction
# Mike Bernico <mike.bernico@gmail.com>

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import numpy as np
import os
from keras.callbacks import TensorBoard


def read_data():
    df = pd.read_csv("./data/bitcoin.csv", index_col=False)
    df["Time"] = pd.to_datetime(df["Timestamp"], unit='s')
    df.index = df["Time"]
    df = df.drop(["Time", "Timestamp"], axis=1)
    return df


def select_dates(df, start, end):
    mask = (df.index > start) & (df.index <= end)
    return df[mask]


def scale_data(df, scaler=None):
    scaled_df = pd.DataFrame(index=df.index)
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_df["Price"] = scaler.fit_transform(df.Close.values.reshape(-1,1))
    return scaler, scaled_df


def diff_data(df):
    df_diffed = df.diff()
    df_diffed.fillna(0, inplace=True)
    return df_diffed


def lag_dataframe(data, lags=1):
    """
    creates shifted lag columns for a dataframe
    e.g. for a df with col1, it creates n lagged versions of col1

    """
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(lags, 0, -1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)

    cols = df.columns.tolist()
    for i, col in enumerate(cols):
        if i == 0:
            cols[i] = "x"
        else:
            cols[i] = "x-" + str(i)

    cols[-1] = "y"
    df.columns = cols
    return df


def prep_data(df_train, df_test, lags):
    df_train = diff_data(df_train)
    scaler, df_train = scale_data(df_train)
    df_test = diff_data(df_test)
    scaler, df_test = scale_data(df_test, scaler)
    df_train = lag_dataframe(df_train, lags=lags)
    df_test = lag_dataframe(df_test, lags=lags)

    X_train = df_train.drop("y", axis=1)
    y_train = df_train.y
    X_test = df_test.drop("y", axis=1)
    y_test = df_test.y

    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test


def build_network(sequence_length=10, batch_shape=100, input_dim=1):
    inputs = Input(batch_shape=(batch_shape, sequence_length, input_dim), name="input")
    lstm1 = LSTM(100, activation='tanh', return_sequences=True, stateful=True, name='lstm1')(inputs)
    lstm2 = LSTM(100, activation='tanh', return_sequences=False, stateful=True, name='lstm2')(lstm1)
    output = Dense(1, activation='tanh', name='output')(lstm2)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model


def create_callbacks(name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log", name), write_graph=True, write_grads=False)

    return [tensorboard_callback]


def make_prediction(X_test, model):
    y_hat = model.predict(X_test)
    y_hat = np.reshape(y_hat, (y_hat.size,))
    return y_hat


def main():
    LAGS=10
    df = read_data()
    df_train = select_dates(df, start="2017-01-01", end="2017-05-31")
    df_test = select_dates(df, start="2017-06-01", end="2017-06-30")
    X_train, X_test, y_train, y_test = prep_data(df_train, df_test, lags=LAGS)
    model = build_network(sequence_length=LAGS)
    callbacks = create_callbacks("lstm_100_100")
    model.fit(x=X_train, y=y_train,
              batch_size=100,
              epochs=10,
              callbacks=callbacks)
    model.save("lstm_model.h5")


if __name__ == "__main__":
    main()
