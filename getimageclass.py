# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:37:15 2016

@author: bigdata
"""

#Import Keras related libraries


#import other python related libraries
import numpy as np

from PIL import Image
from numpy import *
import json
import os

from sklearn.preprocessing import LabelEncoder



model_location="/home/bigdata/spyworkspace/MedicalImageAnalysis/3class_model_wt.h5"
image_location="/home/bigdata/spyworkspace/MedicalImageAnalysis/singlefolder/BRAINIX-0001-0001.dcm.png"
temp_location="/home/bigdata/spyworkspace/MedicalImageAnalysis/temp"

# number of channels
img_channels = 1

img_rows, img_cols = 256, 256

number_classes=3

category_list=['BRAINIX', 'INCISIX', 'PHENIX']
encoder = LabelEncoder()
encoder.fit(category_list)

print(encoder)



#flatten the image before predictions
image_name=os.path.basename(image_location)
if not os.path.exists(temp_location):
    os.makedirs(temp_location)
    
im=Image.open(image_location)
flattened_image_loc=os.path.join(temp_location,image_name)
img = im.resize((img_rows,img_cols))
gray = img.convert('L')
gray.save(flattened_image_loc, "PNG")


immatrix=array([array(Image.open(flattened_image_loc)).flatten()],'f')


train_data = [immatrix]  

X_test = train_data[0]

#print(X_test.shape[0], 'test samples')

X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_test = X_test.astype('float32')

X_test /= 255


# convert class vectors to binary class matrices

#i = 100
#plt.imshow(X_train[i, 0], interpolation='nearest')
#print("label : ", Y_train[i,:])

model=load_model(model_location,number_classes)
img_class=model.predict_classes(X_test)
#print(img_class[0])

#print(encoder.inverse_transform(img_class)[0] + '   ' +str(img_class[0]) )
data = {}
data['Predicted Class'] = encoder.inverse_transform(img_class)[0] + '   ' +str(img_class[0])

probabilities_date={}
img_class_prob=model.predict_proba(X_test).tolist()
for i in xrange(number_classes):
    #print(str(img_class_prob[0][i])+'  '+encoder.inverse_transform(i))
    probabilities_date[encoder.inverse_transform(i)]=str(img_class_prob[0][i])

data['Probabilities']= probabilities_date    
json_data = json.dumps(data,indent=4, sort_keys=True)    

print(json_data)

os.remove(flattened_image_loc)


