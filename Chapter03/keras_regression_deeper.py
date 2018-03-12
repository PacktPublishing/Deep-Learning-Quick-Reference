# Deep Learning Quick Reference Chapter 3: TensorBoard
# Mike Bernico <mike.bernico@gmail.com>

# random seed setting for reproducibility
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from matplotlib import pyplot as plt

from keras.callbacks import TensorBoard

TRAIN_DATA = "./data/train/train_data.csv"
VAL_DATA = "./data/val/val_data.csv"
TEST_DATA = "./data/test/test_data.csv"

def load_data():
    """Loads train, val, and test datasets from disk"""
    train = pd.read_csv(TRAIN_DATA)
    val = pd.read_csv(VAL_DATA)
    test = pd.read_csv(TEST_DATA)

    # we will use sklearn's StandardScaler to scale our data to 0 mean, unit variance.
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)
    # we will use a dict to keep all this data tidy.
    data = dict()

    data["train_y"] = train[:, 10]
    data["train_X"] = train[:, 0:9]
    data["val_y"] = val[:, 10]
    data["val_X"] = val[:, 0:9]
    data["test_y"] = test[:, 10]
    data["test_X"] = test[:, 0:9]
    # it's a good idea to keep the scaler (or at least the mean/variance) so we can unscale predictions
    data["scaler"] = scaler
    return data

def build_network(input_features=None):
    # first we specify an input layer, with a shape == features
    inputs = Input(shape=(input_features,), name="input")
    x = Dense(32, activation='relu', name="hidden1")(inputs)
    x = Dense(32, activation='relu', name="hidden2")(x)
    x = Dense(32, activation='relu', name="hidden3")(x)
    x = Dense(32, activation='relu', name="hidden4")(x)
    x = Dense(16, activation='relu', name="hidden5")(x)
    # for regression we will use a single neuron with linear (no) activation
    prediction = Dense(1, activation='linear', name="final")(x)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


def create_callbacks():
    tensorboard_callback = TensorBoard(log_dir='./ch3_tb_log/dnn', histogram_freq=1, batch_size=32,
                                       write_graph=True, write_grads=False)

    return [tensorboard_callback]


def main():
    data = load_data()

    # load callbacks
    callbacks = create_callbacks()

    # Network with single 32 neuron hidden layer...
    input_features = data["train_X"].shape[1]
    model = build_network(input_features=input_features)
    print("Network Structure")
    print(model.summary())
    print("Training Data Shape: " + str(data["train_X"].shape))
    model.fit(x=data["train_X"], y=data["train_y"], batch_size=32, epochs=500, verbose=1,
              validation_data=(data["val_X"], data["val_y"]), callbacks=callbacks)

    # an example of saving the model for later use
    model.save("regression_model.h5")
    # use model = model.load("regression_model.h5") to load it

    print("Model Train MAE: " + str(mean_absolute_error(data["train_y"], model.predict(data["train_X"]))))
    print("Model Val MAE: " + str(mean_absolute_error(data["val_y"], model.predict(data["val_X"]))))
    print("Model Test MAE: " + str(mean_absolute_error(data["test_y"], model.predict(data["test_X"]))))

    plt.title("Predicted Distribution vs. Actual")
    y_hat = model.predict(data["test_X"])
    sns.distplot(y_hat.flatten(), label="y_hat")
    sns.distplot(data["test_y"], label="y_true")
    plt.legend()
    plt.savefig("pred_dist_deep.png" )



if __name__ == "__main__":
    main()

