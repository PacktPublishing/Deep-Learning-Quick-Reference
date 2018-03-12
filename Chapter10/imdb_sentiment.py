'''
Forked/Refactored from https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py


'''
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, LSTM, Input
from keras.models import Model
from keras.datasets import imdb
from keras.callbacks import TensorBoard, ModelCheckpoint
import os


def load_data(vocab_size):
    data = dict()
    data["vocab_size"] = vocab_size
    (data["X_train"], data["y_train"]), (data["X_test"], data["y_test"]) = imdb.load_data(num_words=vocab_size)
    return data


def pad_sequences(data):
    data["X_train"] = sequence.pad_sequences(data["X_train"])
    data["sequence_length"] = data["X_train"].shape[1]
    data["X_test"] = sequence.pad_sequences(data["X_test"], maxlen=data["sequence_length"])
    return data


def build_network(vocab_size, embedding_dim, sequence_length):
    input = Input(shape=(sequence_length,), name="Input")
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length,
                          name="embedding")(input)
    lstm1 = LSTM(10, activation='tanh', return_sequences=False,
                 dropout=0.2, recurrent_dropout=0.2, name='lstm1')(embedding)
    output = Dense(1, activation='sigmoid', name='sigmoid')(lstm1)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_callbacks(name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log_sentiment", name), write_graph=True,
                                       write_grads=False)
    checkpoint_callback = ModelCheckpoint(filepath="./model-weights" + name + ".{epoch:02d}-{val_loss:.6f}.hdf5",
                                          monitor='val_loss', verbose=0, save_best_only=True)
    return [tensorboard_callback, checkpoint_callback]


def main():

    data = load_data(20000)
    data = pad_sequences(data)
    model = build_network(vocab_size=data["vocab_size"],
                          embedding_dim=100,
                          sequence_length=data["sequence_length"])

    callbacks = create_callbacks("sentiment")

    model.fit(x=data["X_train"], y=data["y_train"],
              batch_size=32,
              epochs=10,
              validation_data=(data["X_test"], data["y_test"]),
              callbacks=callbacks)

    model.save("sentiment.h5")

    score, acc = model.evaluate(data["X_test"], data["y_test"],
                                batch_size=32)
    print('Test loss:', score)
    print('Test accuracy:', acc)

if __name__ == "__main__":
    main()
