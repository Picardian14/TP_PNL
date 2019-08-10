# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:19:05 2019

@author: Santiago
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, LSTM

texts = []
labels = []
def read_data(file_to_read,label_index):
    with open(file_to_read, 'r') as f:
        for line in f:
            data = line.split('|')
            texts.append(data[0])
            labels.append(label_index[data[2]])
    return texts, labels


def model_A(max_words,max_len):
    model = Sequential()
    model.add(Embedding(max_words, 32, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def model_B(max_words,max_len):
    model = Sequential()
    model.add(Embedding(max_words,32,input_length=max_len))
    model.add(LSTM(32, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',dropout=0.0, recurrent_dropout=0.1))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def model_C(max_words,max_len):
    model = Sequential()
    model.add(Embedding(max_words,64,input_length=max_len))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model