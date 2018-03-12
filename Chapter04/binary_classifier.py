# Deep Learning Quick Reference Chapter 4: Using  Deep Learning To Solve Binary Classification  Problems
# Mike Bernico <mike.bernico@gmail.com>

# random seed setting for reproducibility
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from roc_callback import RocAUCScore


TRAIN_DATA = "./data/train/train_data.csv"
VAL_DATA = "./data/val/val_data.csv"
TEST_DATA = "./data/test/test_data.csv"


def load_data():
    """Loads train, val, and test datasets from disk"""
    train = pd.read_csv(TRAIN_DATA)
    val = pd.read_csv(VAL_DATA)
    test = pd.read_csv(TEST_DATA)

    # we will use a dict to keep all this data tidy.
    data = dict()
    data["train_y"] = train.pop('y')
    data["val_y"] = val.pop('y')
    data["test_y"] = test.pop('y')

    # we will use sklearn's StandardScaler to scale our data to 0 mean, unit variance.
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    data["train_X"] = train
    data["val_X"] = val
    data["test_X"] = test
    # it's a good idea to keep the scaler (or at least the mean/variance) so we can unscale predictions
    data["scaler"] = scaler
    return data


def build_network(input_features=None):
    # first we specify an input layer, with a shape == features
    inputs = Input(shape=(input_features,), name="input")
    x = Dense(128, activation='relu', name="hidden1")(inputs)
    x = Dense(64, activation='relu', name="hidden2")(x)
    x = Dense(64, activation='relu', name="hidden3")(x)
    x = Dense(32, activation='relu', name="hidden4")(x)
    x = Dense(16, activation='relu', name="hidden5")(x)
    prediction = Dense(1, activation='sigmoid', name="final")(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    return model


def create_callbacks(data):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log", "5h_adam_20epochs"), histogram_freq=1, batch_size=32,
                                       write_graph=True, write_grads=False)

    roc_auc_callback = RocAUCScore(training_data=(data["train_X"], data["train_y"]),
                                   validation_data=(data["val_X"], data["val_y"]))

    checkpoint_callback = ModelCheckpoint(filepath="./model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc',
                                          verbose=1, save_best_only=True)

    return [tensorboard_callback, roc_auc_callback, checkpoint_callback]


def class_from_prob(x, operating_point=0.5):
    x[x >= operating_point] = 1
    x[x < operating_point] = 0
    return x


def main():
    data = load_data()
    callbacks = create_callbacks(data)
    print("Data Loaded...")
    print("Train Shape X:" + str(data["train_X"].shape)+ " y: "+str(data["train_y"].shape))

    input_features = data["train_X"].shape[1]
    model = build_network(input_features=input_features)
    print("Network Structure")
    print(model.summary())
    model.fit(x=data["train_X"], y=data["train_y"], batch_size=32, epochs=20, verbose=1,
              validation_data=(data["val_X"], data["val_y"]), callbacks=callbacks)

    y_prob_train = model.predict(data["train_X"])
    y_hat_train = class_from_prob(y_prob_train)
    y_prob_val = model.predict(data["val_X"])
    y_hat_val = class_from_prob(y_prob_val)

    print("Model Train Accuracy: " + str(accuracy_score(data["train_y"], y_hat_train)))
    print("Model Val Accuracy: " + str(accuracy_score(data["val_y"], y_hat_val)))
    print("Val ROC: " + str(roc_auc_score(data["val_y"], y_prob_val)))
    print("Val Classification Report")
    print(classification_report(data["val_y"], y_hat_val))


if __name__ == "__main__":
    main()

