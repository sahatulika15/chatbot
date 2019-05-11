"""
Read the DQN1 once and use that for the time being
"""

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.optimizers import Adam
from keras import optimizers
from keras.models import load_model
import time

from .SumTree import SumTree

import sys, os
sys.path.insert(0, os.path.abspath('..'))

class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    print("in memeory")
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class DQNAgent:
    def __init__(self, state_size, action_size, hiddenLayers=[], dropout=0.1, activation='relu', loadname=None,
                 saveIn=False, learningRate=0.01, discountFactor=0.9,
                 epsilon=None):  # the file to load is provided if requirede
        # saveIn is providded if to store the model in from which we load it
        print("in init")
        self.state_size = state_size
        self.action_size = action_size
        #self.memory = deque(maxlen=100000)
	self.memory1 = deque(maxlen=100)
        self.gamma = discountFactor  # discount rate
        if epsilon is None:
            self.epsilon = 1.0  # exploration rate
        else:
            self.epsilon = epsilon
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.00005
        self.learning_rate = learningRate
        # self.noHiddenLayers = noHiddenLayers

        self.hiddenLayers = hiddenLayers
        self.dropout = dropout
        self.activation = activation
        self.model = self._build_model(self.hiddenLayers, self.dropout, self.activation)
	self.model_ = self._build_model(self.hiddenLayers, self.dropout, self.activation)
        self.loadname = loadname
        self.saveInLoaded = saveIn
	self.memory = Memory(200000)
        self.iter = 0  # the amount of model runs and iterations and runs
        if self.loadname is not None:
            self.load(self.loadname)

    def init_record(self):
        pass
        # f = open(self.filename,'w')
        # f.write('The headers for records separated by commma')
        # f.write('The initial state')
        # f.close()

    def update_record(self):
        # f = open(self.filename, 'a')
        # f.write('write the current state')
        # f.close()
        pass

    def _build_model(self, hiddenLayers, dropout, activation):
        # Neural Net for Deep-Q learning Model
        print("in build_model")
        bias = True
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.action_size, inputs_shape=(self.state_size,), kernel_initializer='lecun_uniform',
                            use_bias=bias))
            model.add(Activation("linear"))
        else:
            model.add(Dense(hiddenLayers[0], input_shape=(self.state_size,), kernel_initializer="lecun_uniform",
                            use_bias=bias))
            model.add(Activation(activation))

        for index in range(1, len(hiddenLayers)):
            layerSize = hiddenLayers[index]
            model.add(Dense(layerSize, kernel_initializer="lecun_uniform", use_bias=bias))
            model.add(Activation(activation))
            if dropout > 0:
                model.add(Dropout(dropout))
        model.add(Dense(self.action_size, kernel_initializer="lecun_uniform", use_bias=bias))
        model.add(Activation("linear"))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        optimizer = optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def observe(self, sample):  # in (s, a, r, s_) format
	print(sample)
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.iter % 500 == 0:
            self.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.iter += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

    def _getTargets(self, batch):
	print("in targets")
        no_state = np.zeros(self.state_size)

        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])
	#print(states)
	states = np.reshape(states, (1,5))
	states_ = np.reshape(states_, (1,5))
	#print(states.shape)
        p = self.predict(states)

        p_ = self.predict(states_, target=False)
        pTarget_ = self.predict(states_, target=True)

        x = np.zeros((len(batch), 5))
        y = np.zeros((len(batch), 13))
        errors = np.zeros(len(batch))
        #print(x.shape)
	#print(y.shape)
	#print(errors.shape)
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * pTarget_[i][ np.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)


    def avg_rew(self):
	sum1=0
	i=0
	for elem in self.memory1:
		i=i+1
		sum1=sum1+elem
	avr=sum1/i
        #print(i)
        #print(avr)
	return avr

    def rem_rew(self, reward):
        self.memory1.append((reward))

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def act(self, state, all_act):
        # print("in act")
        if np.random.rand() <= self.epsilon:
            return random.choice(all_act)
        act_values = self.model.predict(state)
        max_key = all_act[0]
        max_val = act_values[0, max_key]
        l = len(all_act)
        for i in range(1, l):
            # print(i)
            k = all_act[i]
            # print(k)
            val_n = act_values[0, k]
            if (val_n > max_val):
                max_val = val_n
                max_key = k
        # print(max_key)
        return max_key  # returns action

    def getTargets(self, batch):
	print("in new targets")
        no_state = np.zeros(self.state_size)

        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])
	#print(states)
	#states = np.reshape(states, (1,5))
	#states_ = np.reshape(states_, (1,23))
	#print(states.shape)
	p=np.zeros((32, 13))
	#print(p.shape)
	p_=np.zeros((32, 13))
	pTarget_=np.zeros((32, 13))
	for i in range(0,len(states)):
        	f = self.predict(states[i])
		p[i]=f
	#print(p)
	for i in range(0,len(states_)):
        	f = self.predict(states_[i],target=False)
		p_[i]=f
        #p_ = self.predict(states_, target=False)
	for i in range(0,len(states_)):
        	f = self.predict(states_[i],target=True)
		pTarget_[i]=f

        #pTarget_ = self.predict(states_, target=True)

        x = np.zeros((len(batch), 5))
        y = np.zeros((len(batch), 13))
        errors = np.zeros(len(batch))
        #print(x.shape)
	#print(y.shape)
	#print(errors.shape)
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * pTarget_[i][ np.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self): 
	BATCH_SIZE = 32   
        batch = self.memory.sample(BATCH_SIZE)
	#print(batch)
        x, y, errors = self.getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.model.fit(x, y, batch_size=32, epochs=1, verbose=0)

    def load(self, name):
        print("in load")
        print("Loading the model {} ".format(name))
        self.model = load_model(name)
        # time.sleep(3)
        # self.model.load_weights(name)

    def save(self, name):
        # saveIn tells us
        print("saving in .... {}".format(name))
        if (self.loadname is None) or (self.saveInLoaded is False):
            print ("Saving in without loading : {}".format(name))
            # self.model.save_weights(name)
            self.model.save(name)
        elif self.saveInLoaded is True and self.loadname is not None:
            print( "Saving in : {}".format(self.loadname))
            # self.model.save_weights(self.loadname)
            self.model.save(self.loadname)
        else:
            print("Error in saving no Conition mathcing")
