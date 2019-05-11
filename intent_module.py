import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
import re
import tensorflow as tf
import itertools
from collections import Counter
from new_utils import *
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, LSTM
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model

# first load all the text and get the sequence lenfgth


class Intent:
    def __init__(self,filename = "intent_module.hdf5",load = True):
        self.sequence_length = 46
        with open('lbl_dict_multiclass.pkl','rb') as f:
            self.lbl_dict=pickle.load(f)
        f.close()
        with open("./New_Tokenizer.tkn",'rb') as f:
        # print("Going to read the file")
            tokenizer=pickle.load(f)
        f.close()
        self.word_index=tokenizer.word_index
        self.embedding_dim = 300
        with open("./Emb_Mat.mat",'rb') as f:
        # print("Going to read the file")
            self.embedding_matrix=pickle.load(f)
        f.close()
        self.model = self.create_model()
        self.filename = filename
        if load:
            self.load_model_weights(filename)

    def load_model_weights(self,filename):
        print ("Loading the Intent model")
        self.model.load_weights(filename)

    def create_model(self):
       model = Sequential()
       model.add(Embedding(input_dim=len(self.word_index)+1, output_dim=self.embedding_dim, weights=[self.embedding_matrix], input_length=self.sequence_length))
       model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='tanh'))
       model.add(MaxPooling1D(pool_size=2))
       #model.add(Flatten())
       model.add(LSTM(100))
       model.add(Dense((len(self.lbl_dict)), activation=(tf.nn.softmax)))
#checkpoint = ModelCheckpoint('./CNN/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='max')
       adam = Adam(lr=0.01, decay=0.0)
#model_loss = custom_loss()
       model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
       return model
       
    def intent_module(self, string):
        train = [string]
        self.sequence_length
        tokenizer = []
        embedding_matrix = []
        with open("./New_Tokenizer.tkn",'rb') as f:
        # print("Going to read the file")
            tokenizer=pickle.load(f)
        f.close()
        
    # embedding_matrix = []
    # with open('./Emb_Mat.mat','rb') as f:
    #     embedding_matrix=pickle.load(f)
    # create the sentence into teh embedding format
        X_train=load_create_padded_data(X_train=train,savetokenizer=False,isPaddingDone=False,maxlen=self.sequence_length,tokenizer_path='./New_Tokenizer.tkn')
        #print("{}".format(X_train))
    #model = 
        #model = load_model("intent_module.h5")
        predict = self.model.predict(X_train)
        pos = predict.argmax()
        #print(self.lbl_dict)
        predict = list(self.lbl_dict.keys())[list(self.lbl_dict.values()).index(pos)]
        return predict

#test=[]
#train=[]
'''
train_Text='./ATIS_TRAIN.txt'
test_Text='./ATIS_TEST.txt'

intent_file = 'cnn_lstm_multiintent.h5'

with open(train_Text) as f:
    for line in f:
    	train.append(line)

with open(test_Text) as f:
    for line in f:
        test.append(line)

text = train + test
text = [s.split(" ") for s in text]
sequence_length = max(len(x) for x in text) # somehow this over here is coming to 47 , whereas the value is supposed to be 46
'''
#sequence_length = 46
'''
Y_train=[]
train_labels='./ATIS_TRAIN_LABEL.txt'


with open(train_labels) as f:
    for line in f:
    	Y_train.append(line)

# this will give us the max sequqnce length of the text

lbl_dict={}
index=0
for dial_lbls in Y_train:
	if dial_lbls not in lbl_dict:
		lbl_dict[dial_lbls]=index
		index=index+1
'''

#f=open('lbl_dict_multiclass.pkl','rb')
#lbl_dict=pickle.load(f)
#f.close()

#print(f"{lbl_dict}\nThe Total number of intents are : {len(lbl_dict)}\n\n")
# get the label dictionaries


if __name__ == "__main__":
    print("{}".format(intent_module("Show me flights")))
