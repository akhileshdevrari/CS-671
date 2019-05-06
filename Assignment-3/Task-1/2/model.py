import numpy as np
import cv2
import pickle
import tensorflow as tf
from keras.applications import VGG16, VGG19
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array

def NEW_MODEL():

    return model
NEW_IMG_SIZE = (224, 224, 3)
name=np.load('fname.npy')
bbox=np.load('fbbox.npy')
print(name.shape, bbox.shape)
BS=12
def data_generator(BS=32):
    while True:
        batches = int(len(name)/BS)
        for i in range(batches):
            data=[]
            for j in range(BS):
                image = cv2.imread(name[i*BS+j])
                image = cv2.resize(image,(NEW_IMG_SIZE[1], NEW_IMG_SIZE[0]))
                image = img_to_array(image)
                image /= 255
                data.append(image)
            data = np.array(data)
            # print(data.shape,bbox[i*BS:(i+1)*BS].shape,bbox[i*BS:(i+1)*BS][:,:,0].shape,bbox[i*BS:(i+1)*BS][1].shape)
            yield(data, {'reg1':bbox[i*BS:(i+1)*BS][:,[0],:,].reshape(BS,4),'reg2':bbox[i*BS:(i+1)*BS][:,[1],:,].reshape(BS,4),'reg3':bbox[i*BS:(i+1)*BS][:,[2],:,].reshape(BS,4),'reg4':bbox[i*BS:(i+1)*BS][:,[3],:,].reshape(BS,4)})

model=VGG16(include_top=False,  weights=None, input_shape=(224,224,3))
output=model.output
reg = Dropout(0.5)(output)
reg = Flatten()(reg)
reg = Dense(4096, activation='relu')(reg)

reg = Dropout(0.2)(reg)
reg = Dense(4096, activation='relu')(reg)

reg = Dropout(0.2)(reg)
reg = Dense(2048, activation='relu')(reg)

reg = Dropout(0.2)(reg)
reg = Dense(512, activation='relu')(reg)
reg2 = Dense(4, activation='linear', name='reg2')(reg)
reg1 = Dense(4, activation='linear', name='reg1')(reg)
reg3 = Dense(4, activation='linear', name='reg3')(reg)
reg4 = Dense(4, activation='linear', name='reg4')(reg)

model = Model(model.input, [reg1,reg2,reg3,reg4])
model.compile(optimizer='Adam', metrics=['accuracy'], loss={'reg1':'mean_squared_error','reg2':'mean_squared_error','reg3':'mean_squared_error','reg4':'mean_squared_error'})
model.summary()
H = model.fit_generator(data_generator(BS),steps_per_epoch=len(name)//BS,epochs=4,verbose=1)
model.save('model_fbbox.h5')
