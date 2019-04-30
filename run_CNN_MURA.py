#  -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = "You Fu"

import numpy as np
np.random.seed(1337)

from scipy.sparse import coo_matrix
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot

import data_loader

###################
# data processing #
###################

im_size = 224  # input image size
X_train_path, Y_train = data_loader.load_path(root_path='/Users/curlyfu/Documents/MURA-6105/MURA-v1.1/train1',
                                              size=im_size)
X_valid_path, Y_valid = data_loader.load_path(root_path='/Users/curlyfu/Documents/MURA-6105/MURA-v1.1/valid',
                                              size=im_size)

from sklearn.model_selection import train_test_split

# x is the feature, and y is the label.
X_train_path, X_test_path, Y_train, Y_test = train_test_split(X_train_path, Y_train, test_size=0.2)

print("loading train set......")
X_train = data_loader.load_image(X_train_path, im_size)  # load trainset
print("loading train set finished")
print("Y_train....")
Y_train = np.asarray(Y_train)
print("Y_train finished")

print("loading valid set......")
X_valid = data_loader.load_image(X_valid_path, im_size)  # loadvalidset
Y_valid = np.asarray(Y_valid)

print("loading test set......")
X_test = data_loader.load_image(X_test_path, im_size)  # loadtest
Y_test = np.asarray(Y_test)

nb_classes = 1
img_dim = (im_size, im_size, 1)  # plus the last dimension, type tuple

###################
# Construct model #
###################
batch_size = 128
nb_classes = 10  #  to_categorical    how many colums
epochs = 1

# number of convolution filters
nb_filters = 64
# size of pooling area for max pooling
pool_size = (3, 3)
# convolution kernel size
kernel_size = (3, 3)

# convert to one_hot type
print("convert Y_train......")
Y_train = np_utils.to_categorical(Y_train, nb_classes)
print("convert Y_test.....")
Y_test = np_utils.to_categorical(Y_test, nb_classes)
print("convert Y_valid......")
Y_valid = np_utils.to_categorical(Y_valid, nb_classes)

# create model
from keras.models import Sequential

model = Sequential()

model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=img_dim))  # Convolution layer 1
model.add(Activation('relu'))  # Activation layer

model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # Convolution layer 2
model.add(Activation('relu'))  # Activation layer     rectified linear unit
model.add(MaxPooling2D(pool_size=pool_size))  # Pooling layer

model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # Convolution layer 3
model.add(Activation('relu'))  # Activation layer
model.add(MaxPooling2D(pool_size=pool_size))  # Pooling layer

model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # Convolution layer 4
model.add(Activation('relu'))  # Activation layer
model.add(MaxPooling2D(pool_size=pool_size))  # Pooling layer

model.add(Dropout(0.25))  # Random inactivation of neurons
model.add(Flatten())  # Pull into one-dimensional data
model.add(Dense(128))  # Fully connected layer 1   Dense layer: A linear operation in which every input is connected to every output by a weight
                        # Convolutional layer: A linear operation using a subset of the weights of a dense layer.
model.add(Activation('relu'))  # Activation layer
model.add(Dropout(0.5))  # Random inactivation of neurons
model.add(Dense(nb_classes))  # Fully connected layer 2
model.add(Activation('softmax'))  # Softmax score


# # kappa score metric
# import keras
# import numpy as np
# import sklearn.metrics as sklm
#
#
# class Metrics(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         # self.confusion = []
#         # self.precision = []
#         # self.recall = []
#         # self.f1s = []
#         self.kappa = []
#         # self.auc = []
#
#     def on_epoch_end(self, epoch, logs={}):
#         # score = np.asarray(self.model.predict(self.validation_data[0]))
#         predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
#         targ = self.validation_data[1]
#
#         # self.auc.append(sklm.roc_auc_score(targ, score))
#         # self.confusion.append(sklm.confusion_matrix(targ, predict))
#         # self.precision.append(sklm.precision_score(targ, predict))
#         # self.recall.append(sklm.recall_score(targ, predict))
#         # self.f1s.append(sklm.f1_score(targ, predict))
#         self.kappa.append(sklm.cohen_kappa_score(targ, predict))
#
#         return


# def kappa_score(y_true, y_pred):

#   assert len(y_true) == len(y_pred), 'Number of examples does not match.'
#   yt = np.asarray(y_true, dtype=int)
#   yp = np.asarray(y_pred, dtype=int)
#   assert np.array_equal(
#       np.unique(yt),
#       [0, 1]), ('Class labels must be binary: %s' % np.unique(yt))
#   observed_agreement = np.true_divide(
#       np.count_nonzero(np.equal(yt, yp)), len(yt))
#   expected_agreement = np.true_divide(
#       np.count_nonzero(yt == 1) * np.count_nonzero(yp == 1) +
#       np.count_nonzero(yt == 0) * np.count_nonzero(yp == 0),
#       len(yt)**2)
#   kappa = np.true_divide(observed_agreement - expected_agreement,
#                          1.0 - expected_agreement)
#   return kappa


###################
#   train model   #
###################
# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics= ['accuracy']# Adam
              )

# train model and fix
# metrics = Metrics()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_valid, Y_valid)),


# predict model
Y_pred = model.predict(X_test)
Y_pred_Array = []
for i in Y_pred:
    j = (i[1]>i[0])*1
    Y_pred_Array.append(j)

Y_test_np = np.asarray(Y_pred_Array)


############################
#   calculate kappa score  #
############################
# cm = sklm.confusion_matrix(Y_test, Y_test_np)
# (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])
from sklearn.metrics import cohen_kappa_score  # confusion_matrix
kappa = cohen_kappa_score(Y_test,Y_test_np)
print (kappa)


#####################
#   evaluate model  #
#####################
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Accuracy score:', score[1])

pyplot.plot(score.history['accuracy'])
pyplot.show()

# dataset
# kappa
# model step conduct train compile and evaluate