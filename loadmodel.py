# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:37:15 2016

@author: bigdata
"""

#Import Keras related libraries
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def load_model(model_location, nb_classes):
    # number of channels
    img_channels = 1
    
    img_rows, img_cols = 256, 256
    
    
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    
    #batch_size to train
    batch_size = 32
    # number of output classes
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    
    model = Sequential()
    
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.load_weights(model_location)
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
    
    return model
    









