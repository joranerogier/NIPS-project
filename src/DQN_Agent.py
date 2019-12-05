from collections import deque
from keras.models import Sequential
from keras.layers import Dense
import random
from keras.optimizers import SGD
import numpy as np

class NeurosmashAgent:
    def __init__(self):
        pass

    """ Agent trained using DQN. """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # replay buffer
        self.gamma = 0.95 # discount factor / decay rate --> SLM Lab  to fit this properly?
        self.epsilon = 1.0 # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=self.state_size))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate)) # originally Adam
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        #for state, action, reward, next_state, done in minibatch:
        for state, action, reward, next_state, done in minibatch:
            #print(f"state: {state}, action: {action}, reward: {reward}, next state: {next_state}, done: {done}")
            if done:
                target = reward # if done
            else:
                # target reward = maximum discounted future reward
                print("not done yet")
                target = (reward + self.gamma * np.amax(self.model.predict(next_state))) # here predict returns 2 values, one for action of going left and one for going right


        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
        return random.randrange(self.action_size)

    def save(self, name):
        self.model.save_weights(name)