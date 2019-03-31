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


########################################################
# Creating images
########################################################

x_train = []
y_train = []
x_test = []
y_test = []

numClass = 0
# Iterating over all attributes of a class
for length in range(0,2):
	for width in range(0,2):
		for color in range(0,2):
			for imgID in range(0,1000):
				# np-array representing data for the image
				data = np.zeros((28, 28, 3), dtype=np.uint8)
				# l = length, w = width of the image to be generated
				l = 7 + 8*length
				w = 1 + 2*width
				# (row,col) if the upper-left corner of the image to be generated
				row = random.randrange(int(5+l/2), int(28-5-l/2-1), 1)
				col = random.randrange(1+5, int(28-l-5), 1)

				# Fill color RGB values in image-np-array
				for i in range(row-1, row+w-1):
					for j in range(col-1, col+l-1):
						data[i][j][2*color] = 255

				# Create image from array
				img = Image.fromarray(data, 'RGB')
				# Upto now we have only created an horizontal line with three class attributes. Now rotate the image to get fourth attribute of class
				for angle in range(0,12):
					tempImg = img.rotate(15*angle)
					if(imgID < 750):
						x_train.append(np.asarray(tempImg, dtype=np.uint8))
						y_train.append(numClass+angle)
					else:
						x_test.append(np.asarray(tempImg, dtype=np.uint8))
						y_test.append(numClass+angle)
						
			numClass = numClass+12


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


# Shuffling x_train and y_train
shuffleArr = []
for i in range(0, len(x_train)):
	shuffleArr.append(i)
random.shuffle(shuffleArr)

x_train_final = []
y_train_final = []

for i in range(0, len(x_train)):
	x_train_final.append(x_train[shuffleArr[i]])
	y_train_final.append(y_train[shuffleArr[i]])

x_train = np.asarray(x_train_final)
y_train = np.asarray(y_train_final)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(np.unique(y_train))


########################################################
# Creating CNN model as per given architecture
########################################################

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 3)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 3)


# y_train = y_train.reshape(y_train.shape[0], 28, 28, 1)
# y_test = y_test.reshape(y_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 3)
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
model.add(Dense(96, activation='softmax'))




# Compiling the model
model.compile(optimizer='adam', 
				loss='sparse_categorical_crossentropy', 
				metrics=['accuracy'])

history = model.fit(x=x_train,y=y_train, epochs=10, validation_split=0.33)


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
np.savetxt('line1Confusion.txt', df_confusion, fmt='%i', delimiter=',')
# f1-score
f1 = f1_score(y_test, y_pred, average=None)
np.savetxt('line1F1.txt', f1, delimiter=',')
# precision-score
precision = precision_score(y_test, y_pred, average=None)
np.savetxt('line1Precision.txt', precision, delimiter=',')
# recall
recall = recall_score(y_test, y_pred, average=None)
np.savetxt('line1Recall.txt', recall, delimiter=',')

print("\n\nTest-Accuracy: ")
print(accuracy_score(y_test, y_pred))

print("\nAverage Precision: ")
print(np.average(precision))

print("\nAverage Recall: ")
print(np.average(recall))

print("\nAverage F1-score: ")
print(np.average(f1))
