import pypianoroll as pn
import numpy as np
import os
from keras.models import load_model
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

seq_len = 10    
model = load_model('stacked.h5')
def pad_along_axis(array, n, axis=0):

    pad_size = (-len(array))%n
    axis_nb = len(array.shape)
    if pad_size < 0:
        return a
    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)
    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
    return b

def trunc_sequences(x_train):
    windows = [] 
    for j in range(len(x_train)):
        temp_x = np.array(np.array_split(pad_along_axis(x_train[j],seq_len),np.ceil(len(x_train[j])/seq_len)))
        windows.append(temp_x)
    return np.array(windows)

x = []
for i in range(1,len(sys.argv)):
    a = pn.parse(sys.argv[i])
    x.append(1*np.sign(a.tracks[0].pianoroll))

x_final = np.array(x)
x_con = trunc_sequences(x_final)
for j in range(len(x_con)):
    bolo = []
    xo = np.expand_dims(x_con[j],axis=1)
    for k in range(len(x_con[j])):
        bolo.extend(model.predict_on_batch(xo[k]))
    model.reset_states()
    bolo = np.concatenate(np.array(bolo))
    result = pn.binarize(pn.Track(pianoroll=bolo*100, program=0, is_drum=False,name='my awesome piano'),threshold=0.05)
    multitracksam11 = pn.Multitrack(tracks=[result])
    pn.write(multitracksam11,'./result/10_' + sys.argv[j+1].split('_')[0] + '.mid')
