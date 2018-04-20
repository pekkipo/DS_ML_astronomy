#We import libraries for linear algebra, graphs, and evaluation of results
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.ndimage.filters import uniform_filter1d
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten,BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from sklearn.metrics import confusion_matrix
from callbacks.confusion_matrix import *
from dal.hdf5dataset import Hdf5Dataset

# LOAD THE DATA
INPUT_LIB = 'data/'
raw_data = np.loadtxt(INPUT_LIB + 'exoTest.csv', skiprows=1, delimiter=',')
x_test = raw_data[:, 1:]
y_test = raw_data[:, 0, np.newaxis] - 1.  # np.newaxis - makes it 2D array, without it I d get just a vector. -1 to make it zeroes and ones instead of 2 and 1
raw_data = np.loadtxt(INPUT_LIB + 'exoTrain.csv', skiprows=1, delimiter=',')
x_train = raw_data[:, 1:]
y_train = raw_data[:, 0, np.newaxis] - 1.
del raw_data


x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1, 1)) / np.std(x_train, axis=1).reshape(-1, 1))
# reshape here because without reshape we ll get 1 row and many columns
# -1 means that numpy should figure out the size in this axis on its own
x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1, 1)) / np.std(x_test, axis=1).reshape(-1, 1))

"""
Subracting mean deviation and dividing by standard deviation is used here for normalization

Subtracting the mean and then dividing by the standard deviation ensures that all of variables have mean zero and variance/standard deviation of 1.

Some statistical methods will treat variables differently depending on their variance and mean. 
 It does not make sense that changing a variable from, say, inches to feet should substantively change the 
 results of the used statistical method. If one normalizes all variables this way, then the results are invariant to the units used.
 https://www.quora.com/How-can-I-know-subtracting-the-mean-and-dividing-by-the-standard-deviation-for-standardizing-is-an-appropriate-method-for-normalizing-data-or-not
"""

# Feature scaling
# from sklearn import preprocessing
# x_train = preprocessing.normalize(x_train)  # return normalized version of X

x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=200)], axis=2)
x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)

model = Sequential()
model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# A bit of data augmentation here
def batch_generator(x_train, y_train, batch_size=32):
    """
    Gives equal number of positive and negative samples, and rotates them randomly in time
    """
    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')

    yes_idx = np.where(y_train[:, 0] == 1.)[0]
    non_idx = np.where(y_train[:, 0] == 0.)[0]

    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)

        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]

        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis=0)

        yield x_batch, y_batch


train_cf_mat = Hdf5Dataset('models/train_cf_matrix.h5', (2, 2))
val_cf_mat = Hdf5Dataset('models/val_cf_matrix.h5', shape=(2, 2))


cf = ConfusionMatrix(2, train_cf_mat.insert, val_cf_mat.insert)

callbacks = [ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs'),
             cf]

metrics = cf.generate_metrics()

#Start with a slightly lower learning rate, to ensure convergence
model.compile(optimizer=Adam(1e-5), loss = 'binary_crossentropy', metrics=['accuracy'] + metrics)
hist = model.fit_generator(batch_generator(x_train, y_train, 32),
                           validation_data=(x_test, y_test),
                           verbose=0, epochs=5,
                           callbacks=callbacks,
                           steps_per_epoch=x_train.shape[1]//32)