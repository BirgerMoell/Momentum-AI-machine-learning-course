import os
import sys

texts = [] # list of text samples
labels_index = {} # dictionary mapping label name to numeric id
labels = [] # list of label ids
for name in sorted(os.listdir('20_newsgroup')):
	path = os.path.join('20_newsgroup', name)
	if os.path.isdir(path):
		label_id = len(labels_index)
		labels_index[name] = label_id
		for fname in sorted(os.listdir(path)):
			if fname.isdigit():
				fpath = os.path.join(path, fname)
				if sys.version_info < (3,):
					f = open(fpath)
				else:
					f = open(fpath, encoding='latin-1')
				t = f.read()
				i = t.find('\n\n') # skip header
				if 0 < i:
					t = t[i:]
				texts.append(t)
				f.close()
				labels.append(label_id)

print('Found %s texts.' % len(texts))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=100)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arrange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = []
f = open(os.path.join(GLOVE_DIR, 'glove.6b.100d.txt'))
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		# words not found in the embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector

from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
					EMBEDDING_DIM,
					weights=[embedding_matrix],
					input_length=MAX_SEQUENCE_LENGTH,
					trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layers(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
		optimizer='rmsprop',
		metrics=['acc'])

#happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
	epochs=2, batch_size=128)


