# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:37:15 2016

@author: bigdata

This is based on few dicom images from http://www.osirix-viewer.com/datasets/

Used following for simplicity: 'BRAINIX', 'INCISIX', 'PHENIX'
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



single_folder_path="/home/bigdata/spyworkspace/MedicalImageAnalysis/singlefolder"
source_folder_path="/home/bigdata/spyworkspace/MedicalImageAnalysis/mripng"
flatten_image_folder="/home/bigdata/spyworkspace/MedicalImageAnalysis/flattened"

# number of channels
img_channels = 1

img_rows, img_cols = 256, 256

number_classes=3

total_categories=sort(os.listdir(source_folder_path))

category_list=total_categories.tolist();

def load_folders(foldername, force=False):
    data_folders= [os.path.join(foldername,folder ) for folder in sorted(os.listdir(foldername))]
    if len(data_folders) !=number_classes:
        raise Exception ('Expected %d folders, one per class. Found %d instead.'
        % (number_classes, len(data_folders)))
    return data_folders
    
cat_folders=load_folders(source_folder_path)


def copy_to_single_folder(sub_folders, single_folder, force=False):
    for folder in sub_folders:
        folder_name=os.path.basename(folder)
        image_files=os.listdir(folder)               
        for image_file in image_files:
            new_img_name=image_file.replace('IM',folder_name)
            shutil.copy(os.path.join(folder,image_file),os.path.join(single_folder_path,new_img_name))
            
            
copy_to_single_folder(cat_folders,single_folder_path)

all_files=sort(os.listdir(single_folder_path))

total_img_count=len(all_files)

label=np.ones((total_img_count),dtype = int)

def generate_label(all_files,label):
    index=0
    for image in all_files:
        img=os.path.basename(image)
        img_class=img[:img.find("-0")]
        label[index]=category_list.index(img_class)
        index+=1
           
    
generate_label(all_files,label)



for file in all_files:
    new_file_path=flatten_image_folder+"/"+os.path.basename(file)
    im=Image.open(single_folder_path+"/"+file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    gray.save(new_file_path, "PNG")


imlist = sort(os.listdir(flatten_image_folder))

immatrix = array([array(Image.open(flatten_image_folder+ "/" + im2)).flatten()
              for im2 in imlist],'f')

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]  


img=immatrix[138].reshape(img_rows,img_cols)
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

(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)






X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
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
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])

#hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#               verbose=1, validation_data=(X_test, Y_test))
           
           
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_split=0.2)
              
model.save_weights("/home/bigdata/spyworkspace/MedicalImageAnalysis/3class_model_wt.h5", True)
#model.save("model_t.h5", true)
''' Random tests on few samples'''
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[110:115]))
print(model.predict_proba(X_test[110:115]))
print(Y_test[110:115])










