'''This is a fork/refactor of https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py

This program trains a newsgroup classifier with a pretrained keras embedding layer. It contrasts
'newsgroup_classifier__word_embeddings.py' which does not use pretrained GloVe vectors.

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split


def load_word_vectors(glove_dir):
    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def load_data(text_data_dir, vocab_size, sequence_length, validation_split=0.2):
    data = dict()
    data["vocab_size"] = vocab_size
    data["sequence_length"] = sequence_length

    # second, prepare text samples and their labels
    print('Processing text dataset')

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)
    print('Found %s texts.' % len(texts))
    data["texts"] = texts
    data["labels"] = labels
    return data


def tokenize_text(data):
    tokenizer = Tokenizer(num_words=data["vocab_size"])
    tokenizer.fit_on_texts(data["texts"])
    data["tokenizer"] = tokenizer
    sequences = tokenizer.texts_to_sequences(data["texts"])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data["X"] = pad_sequences(sequences, maxlen=data["sequence_length"])
    data["y"] = to_categorical(np.asarray(data["labels"]))
    print('Shape of data tensor:', data["X"].shape)
    print('Shape of label tensor:', data["y"].shape)

    # texts and labels aren't needed anymore
    data.pop("texts", None)
    data.pop("labels", None)
    return data


def train_val_test_split(data):

    data["X_train"], X_test_val, data["y_train"],  y_test_val = train_test_split(data["X"], data["y"],
                                                                                 test_size=0.2,
                                                                                 random_state=42)
    data["X_val"], data["X_test"], data["y_val"], data["y_test"] = train_test_split(X_test_val, y_test_val,
                                                                                    test_size=0.25,
                                                                                    random_state=42)
    return data


def embedding_index_to_matrix(embeddings_index, vocab_size, embedding_dim, word_index):
    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(vocab_size, len(word_index))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def build_model(vocab_size, embedding_dim, sequence_length, embedding_matrix):

    sequence_input = Input(shape=(sequence_length,), dtype='int32')
    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                input_length=sequence_length,
                                trainable=False,
                                name="embedding")(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedding_layer)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(20, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def create_callbacks(name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log_newsgroups", name),
                                       write_graph=True,
                                       write_grads=False,
                                       embeddings_freq=1,
                                       embeddings_layer_names="embedding")
    checkpoint_callback = ModelCheckpoint(filepath="./model-weights" + name + ".{epoch:02d}-{val_loss:.6f}.hdf5",
                                          monitor='val_loss', verbose=0, save_best_only=True)
    return [tensorboard_callback, checkpoint_callback]


def main():
    BASE_DIR = './data'
    glove_dir = os.path.join(BASE_DIR, 'glove.6B')
    text_data_dir = os.path.join(BASE_DIR, '20_newsgroup')
    embeddings_index = load_word_vectors(glove_dir)

    data = load_data(text_data_dir, vocab_size=20000, sequence_length=1000)
    data = tokenize_text(data)
    data = train_val_test_split(data)
    data["embedding_dim"] = 100
    data["embedding_matrix"] = embedding_index_to_matrix(embeddings_index=embeddings_index,
                                                         vocab_size=data["vocab_size"],
                                                         embedding_dim=data["embedding_dim"],
                                                         word_index=data["tokenizer"].word_index)

    callbacks = create_callbacks("newsgroups-pretrained")
    model = build_model(vocab_size=data["vocab_size"],
                        embedding_dim=data['embedding_dim'],
                        sequence_length=data['sequence_length'],
                        embedding_matrix=data['embedding_matrix'])

    model.fit(data["X_train"], data["y_train"],
              batch_size=128,
              epochs=10,
              validation_data=(data["X_val"], data["y_val"]),
              callbacks=callbacks)

    model.save("newsgroup_model_word_embedding.h5")

    score, acc = model.evaluate(x=data["X_test"],
                                y=data["y_test"],
                                batch_size=128)
    print('Test loss:', score)
    print('Test accuracy:', acc)

if __name__ == "__main__":
    main()
