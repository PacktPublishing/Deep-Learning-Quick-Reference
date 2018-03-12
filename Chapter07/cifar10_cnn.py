# Deep Learning Quick Reference Chapter 7: Convolutional Neural Networks
# Mike Bernico <mike.bernico@gmail.com>

from keras.datasets import cifar10
from keras.utils import to_categorical, multi_gpu_model
from keras.models import Model
from keras.layers import Dense, Input, Flatten, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report
import os


def load_data():
    # The data, shuffled and split between train and test sets:
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X /= 255
    test_X /= 255

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return {"train_X": train_X, "train_y": train_y,
            "val_X": test_X[:5000, :], "val_y": test_y[:5000, :],
            "test_X": test_X[5000:, :], "test_y": test_y[5000:, :]}


def build_network(num_gpu=1, input_shape=None):
    inputs = Input(shape=input_shape, name="input")

    # convolutional block 1
    conv1 = Conv2D(64, kernel_size=(3,3), activation="relu", name="conv_1")(inputs)
    batch1 = BatchNormalization(name="batch_norm_1")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1")(batch1)

    # convolutional block 2
    conv2 = Conv2D(32, kernel_size=(3,3), activation="relu", name="conv_2")(pool1)
    batch2 = BatchNormalization(name="batch_norm_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2")(batch2)

    # fully connected layers
    flatten = Flatten()(pool2)
    fc1 = Dense(512, activation="relu", name="fc1")(flatten)
    d1 = Dropout(rate=0.2, name="dropout1")(fc1)
    fc2 = Dense(256, activation="relu", name="fc2")(d1)
    d2 = Dropout(rate=0.2, name="dropout2")(fc2)

    # output layer
    output = Dense(10, activation="softmax", name="softmax")(d2)

    # finalize and compile
    model = Model(inputs=inputs, outputs=output)
    if num_gpu > 1:
        model = multi_gpu_model(model, num_gpu)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    return model


def create_callbacks():
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log", "cnn64_cnn32_fc128_fc64_batch_norm"), histogram_freq=1, batch_size=32,
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
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    CHANNELS = 3  # RGB
    data = load_data()
    callbacks = create_callbacks()
    model = build_network(num_gpu=1, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    print(data["train_X"].shape)
    print(data["train_y"].shape)
    print(data["val_X"].shape)
    print(data["val_y"].shape)
    print(data["test_X"].shape)
    print(data["test_y"].shape)
    print(model.summary())

    model.fit(x=data["train_X"], y=data["train_y"],
              batch_size=32,
              epochs=200,
              validation_data=(data["val_X"], data["val_y"]),
              verbose=1,
              callbacks=callbacks)

    print_model_metrics(model, data)


if __name__ == "__main__":
    main()
