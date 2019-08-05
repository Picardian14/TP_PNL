import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense


max_words = 1000
texts = []
labels = []
label_index = { 'contradiction\n': 0, 'entailment\n':1, 'neutral\n':2}

with open('train.txt', 'r') as f:
    for line in f:
        data = line.split('|')
        texts.append(data[0])
        labels.append(label_index[data[2]])
training_samples = len(data)        

with open('val.txt', 'r') as f:
    for line in f:
        data = line.split('|')
        texts.append(data[0])
        labels.append(label_index[data[2]])
validation_samples = len(data) - training_samples

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts) # Creo el índice para los tokens
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

data = pad_sequences(sequences) # devuelve un tensor de shape (num_sequences, maxlen o maxima sequencia)
                                       #Tiene todas las reviews recortadas hasta max_len tokens

labels = np.asarray(labels) #Lo pasa de lista a array ¿por qué?

indices = np.arange(data.shape[0])  #hago un array empezando desde 0 hasta data.shape[0]


x_train = data[:training_samples]
x_val = data[training_samples: training_samples+validation_samples]

y_train = labels[:training_samples]
y_val = labels[training_samples: training_samples+validation_samples]

model = Sequential()



