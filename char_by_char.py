"""
Resource for this code can be found in the following link.
https://blog.devgenius.io/next-word-prediction-using-long-short-term-memory-lstm-13ea21eb9ead

Char by chard word prediction explained clearly when dealing with next word prediction
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
import matplotlib
from numpy.core.multiarray import dtype
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
from pylab import rcParams
# %matplotlib notebook
# %matplotlib inline
from matplotlib.pyplot import figure
matplotlib.use('agg')
np.random.seed(42)

path = 'Dudaktan Kalbe.txt'

# Based on https://blog.devgenius.io/next-word-prediction-using-long-short-term-memory-lstm-13ea21eb9ead
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 12, 5
#Loading the data
text = open(path, "r", encoding='utf-8').read().lower()
print ('Corpus length: ',len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print ("unique chars: ",len(chars))
print(char_indices)
print(indices_char)

SEQUENCE_LENGTH = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i:i+SEQUENCE_LENGTH])
    next_chars.append(text[i+SEQUENCE_LENGTH])
print ('num training examples: ',len(sentences))

X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# By Author
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.05, shuffle = True, random_state = 7)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, shuffle = True, random_state= 7)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_val shape: {}".format(X_val.shape))
print("y val shape: {}".format(y_val.shape))

# Based on https://blog.devgenius.io/next-word-prediction-using-long-short-term-memory-lstm-13ea21eb9ead
model = Sequential();
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=20, shuffle=True, verbose=1, validation_data=(X_val, y_val))

model.save('next_word_model1.h5')
pickle.dump(history, open("history1.p", "wb"))
model = load_model('next_word_model1.h5')
history = pickle.load(open("history1.p", "rb"))

#Based on https://blog.devgenius.io/next-word-prediction-using-long-short-term-memory-lstm-13ea21eb9ead
figure(figsize=(8, 6), dpi=80)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

figure(figsize=(8, 6), dpi=80)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
