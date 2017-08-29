# Visualize training history
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy
import keras

# Remove warnings
#import warnings
#warnings.filterwarnings('ignore')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataset = pd.read_csv("https://www.floydhub.com/viewer/data/res7UHjG5WSPgPStBT84xW/")
# split into input (X) and output 󰀀 variables

# Remove missing values
dataset = dataset.dropna()

# make a label dataset
dataset["Label"] = dataset["Mark"]

# change rest values to
dataset["Mark"][dataset["Mark"] == "REST"] = 0
dataset["Mark"][dataset["Mark"] == "ADDITION"] = 1
dataset["Mark"][dataset["Mark"] == "PASSTHOUGHT"] = 2
dataset["Mark"][dataset["Mark"] == "JUNK"] = 3

# remove the JUNK data
dataset = dataset[dataset.Mark != 2]
dataset = dataset[dataset.Mark != 3]

# shuffle the data
dataset = dataset.sample(frac=1)

X = np.array(dataset.ix[:, :'CH52'])
Y = np.array([[1,0] if i == 0 else [0,1] for i in dataset.Mark])

# Dropout - the number of neurons removed at each layers, who are readded when testing
# Batch size - the number of data points added at each time, affects training time
# Epochs - the number of training/test sessions

# create model
model = Sequential()
model.add(BatchNormalization(input_shape=(52,)))
model.add(Dropout(0.1))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))

model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.5, epochs=250, batch_size=25, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
