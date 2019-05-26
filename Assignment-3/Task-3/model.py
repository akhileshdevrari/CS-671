import keras.layers as KL
from keras.models import Model,Sequential
from keras.optimizers import Adam,SGD
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

from keras.callbacks import Callback

def build_model():
    '''Build model
    '''
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(32, (7, 7), input_shape=(320,480,1),padding='same', kernel_regularizer=l2(0)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(64, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(128, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    
    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(256))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.35))

    # Layer 7
    alexnet.add(Dense(64))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.25))

    # Layer 8
    alexnet.add(Dense(2))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('sigmoid'))
    return alexnet

    
def prepare_model(model):
    '''Compile model
    '''
    model.compile(optimizer=SGD(lr=0.01,decay=0.00001,nesterov=True),loss= 'mean_squared_error', metrics=['mae','mse','accuracy'])

    
def train_model(model, data, label):
    '''
    Train model
    Compile the model

    Return:
    H -- history of training
    '''
    H = model.fit(data,label,validation_split=0.1,epochs=config.EPOCHS,batch_size=config.BATCH_SIZE, callbacks=[WeightsSaver(5)])

    return H
