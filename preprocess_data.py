import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM
import matplotlib.pyplot as plt



max_words = 500
texts = []
labels = []
contradiction = np.asarray([1,0,0])
entailment = np.asarray([0,1,0])
neutral = np.asarray([0,0,1])
label_index = { 'contradiction\n': contradiction, 'entailment\n':entailment, 'neutral\n':neutral}

with open('train.txt', 'r') as f:
    for line in f:
        data = line.split('|')
        texts.append(data[0])
        labels.append(label_index[data[2]])
training_samples = len(texts)        

with open('val.txt', 'r') as f:
    for line in f:
        data = line.split('|')
        texts.append(data[0])
        labels.append(label_index[data[2]])
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

model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_len))
model.add(Flatten())
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val,y_val))

