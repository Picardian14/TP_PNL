# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:44:30 2019

@author: Santiago
"""


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import my_Class
from time import time
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

max_words = 1000
texts = []
labels = []
contradiction = np.asarray([1,0,0])
entailment = np.asarray([0,1,0])
neutral = np.asarray([0,0,1])
label_index = { 'contradiction\n': contradiction, 'entailment\n':entailment, 'neutral\n':neutral}

# leemos archivos que contienen oraciones y etiquetas
texts,labels=my_Class.read_data('val.txt',label_index)
training_samples = len(texts)                
texts,labels=my_Class.read_data('train.txt',label_index)
validation_samples = len(texts) - training_samples

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts) # Creo el índice para los tokens
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

data = pad_sequences(sequences) # devuelve un tensor de shape (num_sequences, maxlen o maxima sequencia)
                                       #Tiene todas las reviews recortadas hasta max_len tokens
max_len = len(data[0]) #ésto da lo que pad_sequences tomo como la maxima longitud de oranciones
labels = np.asarray(labels) #Lo pasa de lista a array ¿por qué?

indices = np.arange(data.shape[0])  #hago un array empezando desde 0 hasta data.shape[0]


x_train = data[:training_samples]
x_val = data[training_samples: training_samples+validation_samples]

y_train = labels[:training_samples]
y_val = labels[training_samples: training_samples+validation_samples]

tensorborad = TensorBoard(log_dir="logs/{}".format(time()))
#model = my_Class.model_A(max_words,max_len)
model = my_Class.model_B(max_words,max_len)

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val,y_val), callbacks=tensorborad)
