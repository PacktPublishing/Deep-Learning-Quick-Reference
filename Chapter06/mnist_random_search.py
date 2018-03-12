from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


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
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return {"batch_size": batches, "optimizer": optimizers, "keep_prob": dropout}


def main():
    data = load_mnist()
    model = KerasClassifier(build_fn=build_network, verbose=0)
    hyperparameters = create_hyperparameters()
    search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=10, n_jobs=1, cv=3,
                              verbose=1)
    search.fit(data["train_X"], data["train_y"])

    print(search.best_params_)

if __name__ == "__main__":
    main()
