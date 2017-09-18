# Visualize training history
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras

# Remove warnings
#import warnings
#warnings.filterwarnings('ignore')

from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataset = pd.read_csv("input/master.csv")
# split into input (X) and output (Y) variables

# remove values we don't want
dataset["Mark"][dataset["Mark"] == "REST"] = 0
dataset["Mark"][dataset["Mark"] == "ADDITION"] = 1
dataset["Mark"][dataset["Mark"] == "PASSTHOUGHT"] = 2
dataset["Mark"][dataset["Mark"] == "JUNK"] = 3

# remove the JUNK data
dataset = dataset[dataset.Mark != 2]
dataset = dataset[dataset.Mark != 3]

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
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#model.add(Dense(100, kernel_initializer='uniform', activation='relu'))

model.add(Dense(2, kernel_initializer='uniform', activation='relu'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.01, epochs=120, batch_size=25, verbose=0)
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
