# Deep Learning Quick Reference Chapter 9: RNNs for time series prediction
# Mike Bernico <mike.bernico@gmail.com>

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from lstm_bitcoin import read_data, select_dates, prep_data
from sklearn.metrics import mean_squared_error


def load_keras_model():
    model = load_model("lstm_model.h5")
    return model


def make_prediction(X_test, model):
    y_hat = model.predict(X_test[0:41700], batch_size=100)
    y_hat = np.reshape(y_hat, (y_hat.size,))
    return y_hat


def calc_rmse(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))


def test_batch(X_test, y_test, batch_size):
    """
    makes test set divisible by batch size for stateful LSTM
    """
    max_val = (X_test.shape[0] // batch_size) * batch_size
    return X_test[0:max_val], y_test[0:max_val]


def main():
    LAGS=10
    df = read_data()
    df_train = select_dates(df, start="2017-01-01", end="2017-05-31")
    df_test = select_dates(df, start="2017-06-01", end="2017-06-30")
    X_train, X_test, y_train, y_test = prep_data(df_train, df_test, lags=LAGS)
    X_test, y_test = test_batch(X_test, y_test, 100)
    model = load_keras_model()
    y_hat = make_prediction(X_test, model)
    y_hat_series = pd.Series(y_hat, index=y_test.index, name="y_hat")
    plt.title("June 2017 Actual vs Predicted")
    y_test.plot()
    pd.Series(y_test.shift(-1), name="y_shifted").plot(alpha=0.7)
    y_hat_series.plot(alpha=0.8)
    plt.legend()
    plt.savefig("prediction_shift.png", bbox_inches='tight')

    print("RMSE = " + str(calc_rmse(y_test, y_hat)))


if __name__ == "__main__":
    main()
