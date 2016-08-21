# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:37:15 2016

@author: bigdata
"""

#Import Keras related libraries
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

#import other python related libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import shutil



flatten_image_folder="/home/bigdata/spyworkspace/MedicalImageAnalysis/flattened"

# number of channels
img_channels = 1

img_rows, img_cols = 256, 256

number_classes=3

category_list=['BRAINIX', 'INCISIX', 'PHENIX']

imlist=sort(os.listdir(flatten_image_folder))

total_img_count=len(imlist)

label=np.ones((total_img_count),dtype = int)

def generate_label(imlist,label):
    index=0
    for image in imlist:
        img=os.path.basename(image)
        img_class=img[:img.find("-0")]
        label[index]=category_list.index(img_class)
        index+=1
           
    
generate_label(imlist,label)



immatrix = array([array(Image.open(flatten_image_folder+ "/" + im2)).flatten()
              for im2 in imlist],'f')

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]  


#img=immatrix[138].reshape(img_rows,img_cols)
#plt.imshow(img)
#plt.imshow(img,cmap='gray')
#print (train_data[0].shape)
#print (train_data[1].shape)


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = number_classes
# number of epochs to train
nb_epoch = 10


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

(X_test, y_test) = (train_data[0],train_data[1])


X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_test = X_test.astype('float32')

X_test /= 255

print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_test = np_utils.to_categorical(y_test, nb_classes)

#i = 100
#plt.imshow(X_train[i, 0], interpolation='nearest')
#print("label : ", Y_train[i,:])

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
model.load_weights("/home/bigdata/spyworkspace/MedicalImageAnalysis/3class_model_wt.h5")

model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])

#hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#               verbose=1, validation_data=(X_test, Y_test))
           
           

              
#model.save("model_t.h5", true)

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[20:23]))
print(model.predict_proba(X_test[20:23]))
print(Y_test[20:23])










