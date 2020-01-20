from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import random
from keras.optimizers import SGD
import numpy as np

class NeurosmashAgent:
    """ Agent trained using DQN. """

    def __init__(self, state_size, action_size, batch_size, weights_path):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.batch_size = batch_size
        self.weights_path = weights_path
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.batch_size, activation='relu', input_dim=self.state_size))
        model.add(Dense(self.batch_size, activation='relu'))
        #model.add(Dense(self.batch_size, activation='relu')) # extra layer
        model.add(Dense(self.batch_size, activation='relu'))
        model.add(Dense(self.batch_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate)) # originally Adam
        model.load_weights(self.weights_path)
        return model

    '''
    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.batch_size, activation='relu', input_dim=self.state_size))
        model.add(Dense(self.batch_size, activation='relu'))
        #model.add(Dense(self.batch_size, activation='relu')) # extra layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate)) # originally Adam
        model.load_weights(self.weights_path)
        return model
    '''

    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
