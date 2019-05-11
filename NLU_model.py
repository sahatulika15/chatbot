import numpy as np
import pickle
import data.data
import data.load 
# from metrics.accuracy import conlleval
import tensorflow as tf
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D
# import progressbar
import re

#print("Loaded Model from disk")

class NLU:
    def __init__(self,filename = "nlu.h5",load = True):
        # loading the data and dictioanries
        #self.w2idx, self.labels2idx,self.train_x,self.train_label,self.val_x,self.val_label=data.data.load_frames()
        #self.w2idx, self.ne2idx, self.labels2idx = self.dicts['words2idx'], self.dicts['tables2idx'], self.dicts['labels2idx']
        self.idx2la={}
        self.w2idx={}

        
        #self.idx2ne = {self.ne2idx[k]: k for k in self.ne2idx}
        #self.idx2la = {self.labels2idx[k]: k for k in self.labels2idx}
        with open("lbl_dict_slot.pkl","rb") as f:
            self.idx2la=pickle.load(f)
        f.close() 
        with open("word_dict_slot.pkl","rb") as f:
            self.w2idx=pickle.load(f)
        f.close()
        self.idx2w = {self.w2idx[k]: k for k in self.w2idx}
        #self.idx2la = {self.labels2idx[k]: k for k in self.labels2idx}
        #print(self.idx2la)
        self.n_classes = len(self.idx2la)
        self.n_vocab = len(self.idx2w)
        self.sequence_length = 46
        self.model = self.create_model()
        self.filename = filename
        if load:
            self.load_model_weights(filename)

    def load_model_weights(self,filename):
        print ("Loading the NLU model")
        self.model.load_weights(filename)

    def create_model(self):
       model = Sequential()
       model.add(Embedding(self.n_vocab,100,input_length=self.sequence_length))
       model.add(Convolution1D(64,4,border_mode='same', activation='relu'))
       model.add(Dropout(0.25))
       model.add(LSTM(100,return_sequences=True))
       model.add(TimeDistributed(Dense(self.n_classes, activation = (tf.nn.softmax))))
       model.compile('rmsprop', 'categorical_crossentropy')
       return model



    def parse_sentence(self,string):
        """
        This will return the various labels for the string
        :param string: The string which is enteree by the user
        :return: return s the labels for each word
        """
        string = string.lower()
        print(string)
        # widx = []
        # for x in string.split(" "):
        #     try:
        #         widx.append(self.w2idx[x])
        #     except:
        #         widx.append('')
        # string = re.sub('\W+', ' ', string)
        widx = [self.w2idx[x] for x in string.split(" ")]
        #print(widx)
        uni = len(widx)
        #print(len(widx))
        #print(uni)
        widx = self.padding(widx, self.sequence_length)
        widx = np.array(widx)
        #print(widx)
        widx = widx[np.newaxis, :]
        pred = self.model.predict_on_batch(widx)
        pred1 = np.argmax(pred, -1)[0]
        #print (pred1)
        # sentence contains the corresponding prediceted labels
        sentence = [self.idx2la[k] for k in pred1]
        #print(sentence)
        sentence = sentence[0:uni]
        #print (string.split(' '))
        print (sentence)
        pred2 = np.argmax(pred, -1)[0]
        #print (pred2)
        #print (sum(pred[0][0]))
        prob_values = [pred[0][i][pred2[i]] for i in range(len(pred2))]
        #print (prob_values)
        prob_values = prob_values[0:uni]

        return [sentence, prob_values]
        # widx contains the
        
        
    def padding(self,train_x,uni):
	#data=train_x+val_x
        #for i in range(0,len(train_x)):
        if(len(train_x)<uni):
            t=uni-len(train_x)
            for p in range(t):
                train_x.append(0)
        return train_x

