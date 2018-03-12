from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dropout
import numpy as np
from hyperband import Hyperband


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


def build_network(keep_prob=0.5, optimizer='adam'):
    inputs = Input(shape=(784,), name="input")
    x = Dense(512, activation='relu', name="hidden1")(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name="hidden2")(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name="hidden3")(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(10, activation='softmax', name="output")(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    return model


def get_params():
    batches = np.random.choice([5, 10, 100])
    optimizers = np.random.choice(['rmsprop', 'adam', 'adadelta'])
    dropout = np.random.choice(np.linspace(0.1, 0.5, 10))
    return {"batch_size": batches, "optimizer": optimizers, "keep_prob": dropout}


def try_params(data, num_iters, hyperparameters):
    model = build_network(keep_prob=hyperparameters["keep_prob"], optimizer=hyperparameters["optimizer"])
    model.fit(x=data["train_X"], y=data["train_y"],
              batch_size=hyperparameters["batch_size"],
              epochs=int(num_iters))
    loss = model.evaluate(x=data["val_X"], y=data["val_y"], verbose=0)

    return {"loss": loss}


def main():
    data = load_mnist()
    hb = Hyperband(data, get_params, try_params)
    results = hb.run()
    print(results)

if __name__ == "__main__":
    main()
