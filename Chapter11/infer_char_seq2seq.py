from keras.models import Model, load_model
from train_char_seq2seq import load_data, one_hot_vectorize
import numpy as np


def load_models():
    model = load_model('char_s2s.h5')
    encoder_model = load_model('char_s2s_encoder.h5')
    decoder_model = load_model('char_s2s_decoder.h5')
    return [model, encoder_model, decoder_model]


def decode_sequence(input_seq, data, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, data['num_decoder_tokens']))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, data['target_token_index']['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = data["reverse_target_char_index"][sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > data['max_decoder_seq_length']):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, data['num_decoder_tokens']))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def create_reverse_indicies(data):
    data['reverse_input_char_index'] = dict(
        (i, char) for char, i in data["input_token_index"].items())
    data['reverse_target_char_index'] = dict(
        (i, char) for char, i in data["target_token_index"].items())
    return data


def main():
    data = load_data()
    data = one_hot_vectorize(data)
    data = create_reverse_indicies(data)
    model, encoder_model, decoder_model = load_models()

    for seq_index in range(100):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = data["encoder_input_data"][seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, data, encoder_model, decoder_model)
        print('-')
        print('Input sentence:', data['input_texts'][seq_index])
        print('Correct Translation:', data['target_texts'][seq_index].strip("\t\n"))
        print('Decoded sentence:', decoded_sentence)

if __name__ == "__main__":
    main()
