import os
import gym
import numpy as np
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# to enable GPU for TensorFlow
ENABLE_GPU = True

# Constant parameters
ENV_NAME = 'CartPole-v0'
NUM_STEPS = 50000               # number of actions to perform
HIDDEN_ACTIVATION = 'relu'      # for multilayer perceptron
OUTPUT_ACTIVATION = 'linear'    # for regession outputs
LEARNING_RATE = 1e-3            # learning ratre

# File paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
INSTANCE_PATH = os.path.join(DIR_PATH, 'logs', ENV_NAME, '{}_steps'.format(NUM_STEPS))
CHECKPOINTS_PATH = os.path.join(INSTANCE_PATH, 'checkpoints')
LOG_FILE_NAME = INSTANCE_PATH + '\\{}_{}_{}_log.json'.format(HIDDEN_ACTIVATION, OUTPUT_ACTIVATION, LEARNING_RATE)
WEIGHT_FILE_NAME = INSTANCE_PATH + '\\{}_{}_{}_weights.h5f'.format(HIDDEN_ACTIVATION, OUTPUT_ACTIVATION, LEARNING_RATE)

def build_callbacks(env_name):
    """This function is used to build the callbacks for DQNAgent training. Including weight/log saving

        # Arguments
            env_name (string): the current environment name

        # Returns
            callbacks (list): a list of callback instances
    """

    checkpoint_weights_filename = CHECKPOINTS_PATH + '\\dqn_' + env_name + '_weights_{step}.h5f'
    log_filename = LOG_FILE_NAME
    # callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=NUM_STEPS)]
    callbacks = [FileLogger(log_filename, interval=500)]
    return callbacks

def build_model(states, actions):
    """This function is used to build the nerual network model for DQNAgent

        # Arguments
            states (list): the current state in terms of:
                [position of cart, velocity of cart, angle of pole, rotation rate of pole]
            actions (list): the avaiable actions:
                [LEFT, RIGHT]

        # Returns
            model: multi-layer network
    """
    
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation=HIDDEN_ACTIVATION))
    model.add(Dense(24, activation=HIDDEN_ACTIVATION))
    model.add(Dense(actions, activation=OUTPUT_ACTIVATION))
    return model

def build_agent(model, actions):
    """This function is used to build the DQNAgent

        # Arguments
            model: multi-layer network
            actions (list): the avaiable actions:
                [LEFT, RIGHT]

        # Returns
            dqn: DQNAgent()
    """
    
    dqn = DQNAgent(model=model, 
                   memory=SequentialMemory(limit=50000, window_length=1), 
                   policy=BoltzmannQPolicy(),
                   nb_actions=actions, 
                   nb_steps_warmup=1000, 
                   target_model_update=1e-2)
    return dqn


if __name__ == "__main__":

    if not ENABLE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not os.path.exists(INSTANCE_PATH):
        os.makedirs(INSTANCE_PATH)
    
    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)

    # initialize environment
    env = gym.make(ENV_NAME)
    states = env.observation_space.shape[0] 
    actions = env.action_space.n
    print(states)

    # create neural network
    model = build_model(states, actions)
    model.summary()
    
    # create DQNAgent
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=LEARNING_RATE), metrics=['mae'])
    
    # fit agent
    callbacks = build_callbacks(ENV_NAME)
    dqn.fit(env, 
            nb_steps=NUM_STEPS, 
            visualize=True, 
            callbacks=callbacks,
            verbose=2
    )
    dqn.save_weights(WEIGHT_FILE_NAME, overwrite=True)
    
    # test agent
    score = dqn.test(env, nb_episodes=10, visualize=True)
    print(np.mean(score.history['episode_reward']))