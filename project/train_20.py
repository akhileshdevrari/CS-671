#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Bidirectional, Reshape, RepeatVector
from keras.layers import LSTM
from keras.layers import Dropout, Activation, Input
from keras.optimizers import Adam,SGD,RMSprop
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pypianoroll as pn,os


# In[38]:


seq_len = 20    


# In[39]:


def prepare_sequences(x_train, y_train, window_length):
    windows = []
    windows_y = [] 
    for j in range(len(x_train)):
        windows_temp = []
        windows_y.append(y_train[j][window_length:])
        for i in range(window_length,len(x_train[j])):
            windows_temp.append(np.ndarray.flatten(x_train[j][i-window_length:i]))
        windows.append(np.array(windows_temp))
    return np.array(windows), np.array(windows_y)


# In[40]:


def pad_along_axis(array, n, axis=0):

    pad_size = (-len(array))%n
    axis_nb = len(array.shape)
    if pad_size < 0:
        return a
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
    return b


# In[41]:


def trunc_sequences(x_train, y_train):
    windows = []
    windows_y = [] 
    for j in range(len(x_train)):
        temp_x = np.array(np.array_split(pad_along_axis(x_train[j],seq_len),np.ceil(len(x_train[j])/seq_len)))
        temp_y = np.array(np.array_split(pad_along_axis(y_train[j],seq_len),np.ceil(len(x_train[j])/seq_len)))
        windows.append(temp_x)
        windows_y.append(temp_y)
    return np.array(windows), np.array(windows_y)


# In[42]:


x_final = np.load('x_trunc.npy',allow_pickle=True)
y_final = np.load('y_trunc.npy',allow_pickle=True)


# In[43]:


x_test = x_final[400:]
y_test = y_final[400:]
x_train = x_final[:400]
y_train = y_final[:400]


# In[45]:


x_con, y_con = trunc_sequences(x_train,y_train)
x_ton, y_ton = trunc_sequences(x_test,y_test)
os.system('rm -r results/')
os.system('mkdir ./results/')


# In[46]:


best = 100.0
model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape = (seq_len,128),recurrent_activation='sigmoid',activation='sigmoid',stateful=True),batch_input_shape = (1,seq_len,128)))
# model.add(Bidirectional(LSTM(128, input_shape = (seq_len,128),recurrent_activation='sigmoid',activation='sigmoid',stateful=True)))
model.add(RepeatVector(seq_len))          
model.add(Bidirectional(LSTM(128,recurrent_activation='sigmoid',activation='sigmoid',return_sequences=True,stateful=True)))         
model.add(TimeDistributed(Dense(128)))
# model.add((Dense(29440)))
# model.add(Dropout(0.2))
# model.add(Reshape((1,230,128)))
model.add(Activation('sigmoid'))
model.compile(optimizer = RMSprop(lr=0.01), loss = 'binary_crossentropy')

for i in range(30):
    print("\nActual Epoch : ",i)
    print('\n')
    for j in range(400):
        print(j)
        xo = np.expand_dims(x_con[j],axis=1)
        yo = np.expand_dims(y_con[j],axis=1)
        for k in range(len(x_con[j])):
            print(model.train_on_batch(xo[k],xo[k]),end=' ')
        model.reset_states()
        print('\n')
    print('\n')
    print('Testing')
    print('\n')
    kolo = []
    for j in range(5):
        print(j)
        bolo = []
        xo = np.expand_dims(x_ton[j],axis=1)
        for k in range(len(x_ton[j])):
            bolo.extend(model.predict_on_batch(xo[k]))
        model.reset_states()
        bolo = np.concatenate(np.array(bolo))
        kolo.append(bolo)
    jojo = np.array(kolo)
    for j in range(5):
        r2 = pn.Track(pianoroll=jojo[1]*300, program=0, is_drum=False,name='my awesome piano')
        multitrack2 = pn.Multitrack(tracks=[r2])
        os.system('mkdir ./results/'+str(i))
        pn.write(multitrack2,'./results/'+str(i)+'/'+str(j)+'.mid')
    model.save('finale.h5')



