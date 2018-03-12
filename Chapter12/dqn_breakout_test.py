import numpy as np
import gym

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, Convolution2D, Permute, Activation
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.processors import Processor
from PIL import Image


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


def build_model(state_size, num_actions):
    input_shape = (4,) + state_size
    model = Sequential()
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))
    print(model.summary())
    return model


def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks


def main():
    ENV_NAME = 'BreakoutDeterministic-v4'
    INPUT_SHAPE = (84, 84)
    WINDOW_LENGTH = 4
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(42)
    env.seed(42)
    num_actions = env.action_space.n

    model = build_model(INPUT_SHAPE, num_actions)
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    callbacks = build_callbacks(ENV_NAME)


    # After training is done, we save the final weights.
    dqn.load_weights('dqn_BreakoutDeterministic-v4_weights_1750000.h5f')

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=10, visualize=True)

if __name__ == '__main__':
    main()

