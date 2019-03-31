from PIL import Image, ImageDraw
import random
import numpy as np
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense
from keras.utils import np_utils


#Downloading MNIST dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


# y_train = np_utils.to_categorical(y_train, 10)

# Creating a Sequential Model and adding the layers as given in assignment Task1

model = Sequential()
# First layer: 7x7 Convolutional Layer with 32 filters and stride of 1
model.add(Conv2D(filters=32, kernel_size=(7,7), strides=1, input_shape=input_shape))
# ReLU Activation Layer
model.add(Activation('relu'))
# Batch Normalization Layer
model.add(BatchNormalization())
# 2x2 Max Pooling layer with a stride of 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
# fully connected layer with 1024 output units
model.add(Dense(1024))
# # ReLU Activation Layer
model.add(Activation('relu'))


# Final output layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compiling the model
model.compile(optimizer='adam', 
				loss='sparse_categorical_crossentropy', 
				metrics=['accuracy'])
history = model.fit(x=x_train,y=y_train, epochs=20, validation_split=0.33)


# plot learning curves

print("\n")
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


Y_pred = model.predict(x_test, 2)
y_pred = np.argmax(Y_pred, axis=1)

# Saving confusion-matrix, recall, precision and f-measure for each class
df_confusion = confusion_matrix(y_test, y_pred).astype(int)
np.savetxt('mnist1Confusion.txt', df_confusion, fmt='%i', delimiter=',')
# f1-score
f1 = f1_score(y_test, y_pred, average=None)
np.savetxt('mnist1F1.txt', f1, delimiter=',')
# precision-score
precision = precision_score(y_test, y_pred, average=None)
np.savetxt('mnist1Precision.txt', precision, delimiter=',')
# recall
recall = recall_score(y_test, y_pred, average=None)
np.savetxt('mnist1Recall.txt', recall, delimiter=',')

print("\n\nTest-Accuracy: ")
print(accuracy_score(y_test, y_pred))

print("\nAverage Precision: ")
print(np.average(precision))

print("\nAverage Recall: ")
print(np.average(recall))

print("\nAverage F1-score: ")
print(np.average(f1))
