# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:19:05 2019

@author: Santiago
"""

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

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
