import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt
from history_plotter import historyPlotter

def toTextsAndLabels (path_to_data):
    labels = [] #Inicializamos lo que serán nuestras listas palabras y etiquetas
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(path_to_data,label_type) #Armo el path para ../neg y ../pos
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt': #agarro todo lo que sea un .txt
                f = open(os.path.join(dir_name, fname)) 
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return texts, labels
    
max_len = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

imdb_dir = '/home/mr-mister/Ivan/LearningDeep/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')#Concatena paths de forma inteligente


texts, labels = toTextsAndLabels(train_dir)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts) # Creo el índice para los tokens
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=max_len) # devuelve un tensor de shape (num_sequences, maxlen o maxima sequencia)
                                       #Tiene todas las reviews recortadas hasta max_len tokens

labels = np.asarray(labels) #Lo pasa de lista a array ¿por qué?

indices = np.arange(data.shape[0])  #hago un array empezando desde 0 hasta data.shape[0]
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
x_val = data[training_samples: training_samples+validation_samples]

y_train = labels[:training_samples]
y_val = labels[training_samples: training_samples+validation_samples]


model = Sequential()

model.add(Embedding(max_words, 32, input_length=max_len))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val,y_val))

historyPlotter(history)

test_dir = os.path.join(imdb_dir, 'test')
test_texts, test_labels = toTextsAndLabels(test_dir)
test_sequences = tokenizer.texts_to_sequences(test_texts)
x_test = pad_sequences(test_sequences, maxlen=max_len)
y_test = np.asarray(test_labels)

print (model.evaluate(x_test,y_test))


