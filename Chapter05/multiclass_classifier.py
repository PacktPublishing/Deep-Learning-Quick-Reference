# Deep Learning Quick Reference Chapter 5: Multiclass Classification
# Mike Bernico <mike.bernico@gmail.com>

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report
import numpy as np
import os



def load_mnist():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(-1, 784)
    test_X = test_X.reshape(-1, 784)
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X /= 255
    test_X /= 255

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return {"train_X": train_X[:55000, :], "train_y": train_y[:55000, :],
            "val_X": train_X[55000:, :], "val_y": train_y[55000:, :], "test_X": test_X, "test_y": test_y}


def build_network(input_features=None):
    # first we specify an input layer, with a shape == features
    inputs = Input(shape=(input_features,), name="input")
    x = Dense(512, activation='relu', name="hidden1")(inputs)
    x = Dense(256, activation='relu', name="hidden2")(x)
    x = Dense(128, activation='relu', name="hidden3")(x)
    prediction = Dense(10, activation='softmax', name="output")(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    return model


def create_callbacks():
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log", "mnist_512_256_128"), histogram_freq=1, batch_size=32,
                                       write_graph=True, write_grads=False)
    checkpoint_callback = ModelCheckpoint(filepath="./model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc',
                                          verbose=0, save_best_only=True)
    return [tensorboard_callback, checkpoint_callback]


def print_model_metrics(model, data):
    loss, accuracy = model.evaluate(x=data["test_X"], y=data["test_y"])
    print("\n model test loss is "+str(loss)+" accuracy is "+str(accuracy))

    y_softmax = model.predict(data["test_X"])  # this is an n x class matrix of probabilities
    y_hat = y_softmax.argmax(axis=-1)  # this will be the class number.
    test_y = data["test_y"].argmax(axis=-1)  # our test data is also categorical
    print(classification_report(test_y, y_hat))


def main():
    data = load_mnist()
    callbacks = create_callbacks()
    model = build_network(data["train_X"].shape[1])
    model.fit(x=data["train_X"], y=data["train_y"],
              batch_size=30,
              epochs=50,
              validation_data=(data["val_X"], data["val_y"]),
              verbose=1,
              callbacks=callbacks)

    print_model_metrics(model, data)


if __name__ == "__main__":
    main()
