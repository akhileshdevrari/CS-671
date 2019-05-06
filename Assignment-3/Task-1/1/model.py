import numpy as np
import cv2
import pickle
import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer

nb_classes = 3

model=VGG16(include_top=False, weights=None, input_shape=(224,224,3))
output=model.output
classification = Dropout(0.5)(output)
classification = Flatten()(classification)
classification = Dense(4096, activation='relu')(classification)
x = classification
classification = Dense(nb_classes, activation='softmax', name='class')(classification)

reg = Dropout(0.2)(x)
reg = Dense(4096, activation='relu')(reg)
reg = Dense(4, activation='linear', name='reg')(reg)

model = Model(model.input, [classification,reg])
model.compile(optimizer='Adam', metrics=['accuracy'], loss={'class':'categorical_crossentropy','reg':'mean_squared_error'})
model.summary()

NEW_IMG_SIZE = (224, 224, 3)
name=np.load('name.npy')
label=np.load('label.npy')
bbox=np.load('bbox.npy')
classLB = LabelBinarizer()
class_label = classLB.fit_transform(label)

BS=64
def data_generator(BS=32):    
    while True:
        batches = int(len(label)/BS)
        for i in range(batches):
            data=[]
            for j in range(BS):
                image = cv2.imread(name[i*BS+j])
                image = cv2.resize(image,(NEW_IMG_SIZE[1], NEW_IMG_SIZE[0]))
                image = img_to_array(image)
                image /= 255
                data.append(image)
            data = np.array(data)
            yield(data, {'class':class_label[i*BS:(i+1)*BS],'reg':bbox[i*BS:(i+1)*BS]})

H = model.fit_generator(data_generator(BS),steps_per_epoch=len(label)//BS,epochs=10,verbose=1)
model.save('model_bbox.h5')
print("[INFO] serializing category label binarizer...")
f = open("classbin", "wb")
f.write(pickle.dumps(classLB))
f.close()
