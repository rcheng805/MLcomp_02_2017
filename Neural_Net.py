import numpy as np
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Activation, Flatten, Dropout
from scipy import sparse

#Helper function to load data
def load_sparse_csr(filename):
        loader = np.load(filename)
        return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                 shape=loader['shape'])

#Load data
X_train = load_sparse_csr('train_data_x.npz')
data_shape = X_train.shape
y_train = np.load('train_data_y.npy')
X_train = X_train.toarray()
X_train = X_train.reshape(data_shape)

## Create your own model here given the constraints in the problem
#act = keras.layers.advanced_activations.LeakyReLU(alpha=0.1)
act = Activation('relu')
model = Sequential()
model.add(Dense(10,input_shape=(X_train.shape[1],)))  # Use np.reshape instead of this in hw
model.add(act)
model.add(Dense(1))
model.add(Activation('softmax'))

## Printing a summary of the layers and weights in your model
model.summary()

#Compile and fit the model
opt = RMSprop(lr=0.1)
model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['accuracy'])

fit = model.fit(X_train, y_train, batch_size=1024, nb_epoch=15,
    verbose=1)
y_prediction = model.predict(X_train)
score = model.evaluate(X_train, y_train, verbose=0)     

#Print and save the results/prediction
print('Test score:', score[0])
print('Test accuracy:', score[1])
np.savetxt('Y_prediction.csv', y_prediction,delimiter=',')
