# Deep Learning Quick Reference Chapter 11: Seq2Seq
# Mike Bernico <mike.bernico@gmail.com>
# This program expects english to french sentance pairs located in chapter_11/data/
# Dataset can be found at http://www.manythings.org/anki/fra-eng.zip

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os
from pathlib import Path


def load_data(num_samples=50000, start_char='\t', end_char='\n', data_path='data/fra-eng/fra.txt'):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    lines = open(Path(data_path), 'r', encoding='utf-8').read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        target_text = start_char + target_text + end_char
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    return {'input_texts': input_texts, 'target_texts': target_texts,
            'input_chars': input_characters, 'target_chars': target_characters,
            'num_encoder_tokens': num_encoder_tokens, 'num_decoder_tokens': num_decoder_tokens,
            'max_encoder_seq_length': max_encoder_seq_length, 'max_decoder_seq_length': max_decoder_seq_length}


def one_hot_vectorize(data):
    input_chars = data['input_chars']
    target_chars = data['target_chars']
    input_texts = data['input_texts']
    target_texts = data['target_texts']
    max_encoder_seq_length = data['max_encoder_seq_length']
    max_decoder_seq_length = data['max_decoder_seq_length']
    num_encoder_tokens = data['num_encoder_tokens']
    num_decoder_tokens = data['num_decoder_tokens']

    input_token_index = dict([(char, i) for i, char in enumerate(input_chars)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_chars)])
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    data['input_token_index'] = input_token_index
    data['target_token_index'] = target_token_index
    data['encoder_input_data'] = encoder_input_data
    data['decoder_input_data'] = decoder_input_data
    data['decoder_target_data'] = decoder_target_data
    return data


def build_models(lstm_units, num_encoder_tokens, num_decoder_tokens):
    # train model
    encoder_input = Input(shape=(None, num_encoder_tokens), name='encoder_input')
    encoder_outputs, state_h, state_c = LSTM(lstm_units, return_state=True, name="encoder_lstm")(encoder_input)
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(None, num_decoder_tokens), name='decoder_input')
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True,
                                 name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='softmax_output')
    decoder_output = decoder_dense(decoder_outputs)

    model = Model([encoder_input, decoder_input], decoder_output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    encoder_model = Model(encoder_input, encoder_states)

    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_input] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def create_callbacks(name):
    tensorboard_callback = TensorBoard(log_dir=os.path.join(os.getcwd(), "tb_log_char_s2s", name),
                                       write_graph=True,
                                       write_grads=False)
    return [tensorboard_callback]


def main():
    data = load_data()
    data = one_hot_vectorize(data)
    callbacks = create_callbacks("char_s2s")
    model, encoder_model, decoder_model = build_models(256, data['num_encoder_tokens'], data['num_decoder_tokens'])
    print(model.summary())

    model.fit(x=[data["encoder_input_data"], data["decoder_input_data"]],
              y=data["decoder_target_data"],
              batch_size=64,
              epochs=100,
              validation_split=0.2,
              callbacks=callbacks)

    model.save('char_s2s_train.h5')
    encoder_model.save('char_s2s_encoder.h5')
    decoder_model.save('char_s2s_decoder.h5')

if __name__ == "__main__":
    main()

