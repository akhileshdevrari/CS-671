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
import pickle
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import keras.backend as K
from keras.models import Model
import cv2
from keras.preprocessing import image
from contextlib import redirect_stdout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Downloading MNIST dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = (x_train[:2000], y_train[:2000]), (x_test[:350], y_test[:350])
# print('[INFO] x_train shape',x_train.shape)

def preProcess(x_train, y_train, x_test, y_test):
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('[INFO] x_train shape:', x_train.shape, x_train.dtype)
    print('[INFO] x_test shape:', x_test.shape, x_train.dtype)
    print('[INFO] Number of images in x_train', x_train.shape[0])
    print('[INFO] Number of images in x_test', x_test.shape[0])
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = preProcess(x_train, y_train, x_test, y_test)

def trainModel():
    # y_train = np_utils.to_categorical(y_train, 10)

    # Creating a Sequential Model and adding the layers as given in assignment Task1
    input_shape = (28, 28, 1)
    model = Sequential()
    # First layer: 7x7 Convolutional Layer with 32 filters and stride of 1
    model.add(Conv2D(filters=32, kernel_size=(5,5), strides=1, input_shape=input_shape))
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
    history = model.fit(x=x_train,y=y_train, epochs=10, validation_split=0.33)
    print(model.summary())
    
    return model, history
def saveModel():
    print('Saving model, history and model summary ')

    with open('model_summary','w') as fi:
        with redirect_stdout(fi):
            model.summary()
    # with open('mnist_histobject','wb') as fi:
    # 	pickle.dump(history.history, fi)

    model.save('mnist.h5')

def loadModel():
    model = load_model('mnist.h5', custom_objects={"tf": tf})
    # plot_model(model, to_file='model_plot.png',show_shapes=True, show_layer_names=True)
    return model

model = loadModel()

# model, history = trainModel()
# saveModel()
layer_outputs = [layer.output for layer in model.layers]


def am(test_image):
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(test_image)
    print('[INFO] activations :', len(layer_outputs), len(activations))
    layer_names = ['conv2d_1', 'activation_1','max_pooling2d_1']
    activ_list = [activations[0], activations[1], activations[3]]

    images_per_row = 16
    grid=[]
    scales=[]
    for layer_name, layer_activation in zip(layer_names, activ_list):
        n_features = layer_activation.shape[-1]
        print('FEATURES : ', n_features)
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 32
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        # plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        # plt.title(layer_name)
        # plt.grid(False)
        grid.append(display_grid)
        scales.append(scale)
        # plt.imshow(display_grid, aspect='auto', cmap='plasma')
        # plt.savefig(layer_name+"_grid.jpg", bbox_inches='tight')
    return layer_names, grid, scales

def hm(test_image, class_index):
    # preds = model.predict(test_image)
    # print ("Predicted: ", preds)
    # print(model.output)
    #985 is the class index for class 'Daisy' in Imagenet dataset on which my model is pre-trained
    flower_output = model.output[:, class_index]
    last_conv_layer = model.get_layer('activation_2')
    grads = K.gradients(flower_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([test_image])

    # #1024 is the number of filters/channels in 'activation_2' layer
    for i in range(1024):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)


    # #Using cv2 to superimpose the heatmap on original image to clearly illustrate activated portion of image
    test_image = test_image.reshape (28, 28)
    test_image = np.uint8(255 * test_image)
    
    heatmap = cv2.resize(heatmap, (28, 28))
    heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + test_image
    
    # cv2.imwrite('h_test.jpg', test_image)
    # cv2.imwrite('h_heatmap.jpg', heatmap)
    # cv2.imwrite('h_superimposed.jpg', superimposed_img)
    
    # plt.imshow(test_image, aspect='auto', cmap='plasma')
    # plt.imshow(heatmap, aspect='auto', cmap='plasma')
    # plt.imshow(superimposed_img, aspect='auto', cmap='plasma')
    
    # plt.savefig("h_test.jpg", bbox_inches='tight')
    # plt.savefig("h_heatmap.jpg", bbox_inches='tight')
    # plt.savefig("h_superimposed.jpg", bbox_inches='tight')

    return test_image, heatmap, superimposed_img

def gd():
    #-------------------------------------------------
    #Utility function for displaying filters as images
    #-------------------------------------------------
    def deprocess_image(x):
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1
        x += 0.5
        x = np.clip(x, 0, 1)
        x *= 255
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    #---------------------------------------------------------------------------------------------------
    #Utility function for generating patterns for given layer starting from empty input image and then 
    #applying Stochastic Gradient Ascent for maximizing the response of particular filter in given layer
    #---------------------------------------------------------------------------------------------------

    def generate_pattern(layer_name, filter_index, size=28):
        layer_output = model.get_layer(layer_name).output
        # print(layer_output)
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.gradients(loss, model.input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([model.input], [loss, grads])
        input_img_data = np.random.random((1, size, size, 1)) * 20 + 128.
        # print(input_img_data)
        step = 1.

        for i in range(80):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
            
        img = input_img_data[0]
        return deprocess_image(img)

    #------------------------------------------------------------------------------------------
    #Generating convolution layer filters for intermediate layers using above utility functions
    #------------------------------------------------------------------------------------------
    layer_name = 'dense_1'
    size = 28
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 1))

    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
            
    # plt.figure(figsize=(20, 20))    
    # plt.savefig(results)
    print(results.shape)
    results = results.reshape (results.shape[0], results.shape[1])
    # cv2.imwrite('results.jpg', results)
    return results
    # plt.imshow(results, aspect='auto', cmap='plasma')
    # plt.savefig("results.jpg", bbox_inches='tight')

for i in range(7,8):
    test_image = x_test[i].reshape(1, 28, 28, 1)
    # img = img.reshape(1, 28, 28, 1)
    class_index = (int)(y_test[i])-1

    print('[INFO] TEST IMAGE', test_image.shape, class_index+1)
    layer_names, grid, scales = am(test_image)
    for layer_name, display_grid, scale in zip(layer_names, grid, scales):
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='plasma')
        plt.savefig(layer_name+'_grid_'+str(i)+'.jpg', bbox_inches='tight')
        plt.close()

    test_image, heatmap, superimposed_img = hm(test_image, class_index)
    # cv2.imwrite('h_test_'+str(i)+'.jpg', test_image)
    # cv2.imwrite('h_heatmap_'+str(i)+'.jpg', heatmap)
    # cv2.imwrite('h_superimposed_'+str(i)+'.jpg', superimposed_img)

    plt.imshow(test_image, aspect='auto', cmap='plasma')
    plt.savefig('h_test_'+str(i)+'.jpg', bbox_inches='tight')
    plt.close()
    
    plt.imshow(heatmap, aspect='auto', cmap='plasma')
    plt.savefig('h_heatmap_'+str(i)+'.jpg', bbox_inches='tight')
    plt.close()
    
    plt.imshow(superimposed_img, aspect='auto', cmap='plasma')
    plt.savefig('h_superimposed_'+str(i)+'.jpg', bbox_inches='tight')
    plt.close()
    
    results = gd()
    # print(results)
    # cv2.imwrite('results_'+str(i)+'.jpg', results)
    plt.imshow(results, aspect='auto', cmap='plasma')
    plt.savefig('results_'+str(i)+'.jpg', bbox_inches='tight')
    plt.close()





