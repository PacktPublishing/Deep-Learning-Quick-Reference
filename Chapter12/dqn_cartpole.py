import numpy as np
import gym

from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger


def build_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model


def build_callbacks(env_name):
    checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks


def main():
    ENV_NAME = 'CartPole-v0'
    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(42)
    env.seed(42)
    num_actions = env.action_space.n
    state_space = env.observation_space.shape[0]
    print(num_actions)

    model = build_model(state_space, num_actions)

    memory = SequentialMemory(limit=50000, window_length=1)

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=10000)

    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    callbacks = build_callbacks(ENV_NAME)

    dqn.fit(env, nb_steps=50000,
            visualize=False,
            verbose=2,
            callbacks=callbacks)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)

if __name__ == '__main__':
    main()

