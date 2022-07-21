import string
import keras.optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from numpy import array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dropout
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

path = "Tez hakkında/Dudaktan Kalbe.txt"

#Based on https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens


# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w', encoding="utf8")
	file.write(data)
	file.close()


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

#Based on https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
doc = load_doc(path)
print(doc[:200])

# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))


# organize into sequences of tokens
length = 30
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = 'dudaktan_kalbe_sequences.txt'
save_doc(sequences, out_filename)
print("saved")

in_filename = 'dudaktan_kalbe_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)


vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index

sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
#y = to_categorical(y, num_classes=vocab_size)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.05, shuffle = True, random_state = 7)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, shuffle = True, random_state= 7)
seq_length = X_train.shape[1]
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_val shape: {}".format(X_val.shape))
print("y val shape: {}".format(y_val.shape))


embedding_index = {}
f = open("Tez hakkında/vectors.txt", encoding="utf8")

#Adapted from: https://www.youtube.com/watch?v=ivqXiW0X42Q
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype = "float32")
	embedding_index[word] = coefs
f.close()
print(len(embedding_index))

embedding_matrix = np.zeros((vocab_size, 300))
for word, i in word_index.items():
	embedding_vector = embedding_index.get(word)
	if i < vocab_size:
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

#Based on https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=seq_length))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

optimizer = keras.optimizers.RMSprop(learning_rate = 0.005)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, epochs=1, shuffle=True, verbose=1, validation_data=(X_val, y_val))
#model.save('model2.h5')
#dump(tokenizer, open('tokenizer.pkl', 'wb'))

#By author
figure(figsize=(8, 6), dpi=80)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# load cleaned text sequences	
in_filename = 'wiki_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1


# load the model
model = load_model('model2.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

# select a seed text
seed_text = ""

# generate new text

generated = generate_seq(model, tokenizer, seq_length, seed_text, 5)


