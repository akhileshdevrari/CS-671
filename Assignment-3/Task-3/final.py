from argparse import ArgumentParser
import numpy as np
import cv2
import glob, os
from os import listdir
from os.path import isfile, join
from model import *
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def go(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    po = 2 * sobelx * sobely
    go = pow((sobely*sobelx),2)
    a = 0.5*np.arctan2(sobely,sobelx)
    oy = cv2.blur(np.sin(a),(5,5))
    ox = cv2.blur(np.cos(a),(5,5))
    angle = sigmoid(np.arctan2(oy,ox))
    return angle

class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights.h5' 
            self.model.save_weights(name)
        self.batch += 1

class Config():
    """Configuration 
    """
    IMG_X = 320
    IMG_Y = 480
    EPOCHS = 100
    BATCH_SIZE = 32
    
config = Config()

parser = ArgumentParser()
parser.add_argument("--phase", dest="phase", help="Testing or Training mode")
parser.add_argument("--epochs", dest="epochs", help="The number of epochs")

args = parser.parse_args()

if(args.phase == "train"):
    path = input("Enter the training folder : ")
    config.EPOCHS = args.epochs
    mypath = path+"/Data/"
    gtpath = path+"/Ground_truth/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    x_data = []
    y_data = []
    for filename in onlyfiles:
        input_data = mypath+filename
        output_data = gtpath+filename.split(".")[0]+"_gt.txt"
        out = np.loadtxt(output_data)
        im = cv2.imread(input_data,cv2.IMREAD_GRAYSCALE)
        tho = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)
        th3 = tho.astype(float)
        th3 = cv2.resize(th3, (320,480))
        out[0] = (out[0]/im.shape[0])
        out[1] = (out[1]/im.shape[1])
        angle = go(th3)
        x_data.append(np.reshape(angle,(320,480,1)))
        y_data.append(out)

    x_final = np.array(x_data)
    y_final = np.array(y_data)
    model = build_model()
    prepare_model(model)

elif(args.phase == "test"):
    path = raw_input("Enter the testing folder : ")
    mypath = path+"/Data/"
    resultpath = path+"/Result/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    x_data = []
    original = []
    for filename in onlyfiles:
        input_data = mypath+filename
        im = cv2.imread(input_data,cv2.IMREAD_GRAYSCALE)
        tho = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)
        th3 = tho.astype(float)
        th3 = cv2.resize(th3, (320,480))
        angle = go(th3)
        original.append(im.shape)
        x_data.append(np.reshape(angle,(320,480,1)))

    x_final = np.array(x_data)
    model = build_model()
    try:
        model.load_weights('nice.h5')
    except:
        model.load_weights('weights.h5')
    prepare_model(model)
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    prediction = model.predict(x_final)
    for i in range(len(prediction)):
        prediction[i][0] = round(prediction[i][0] * original[i][0])
        prediction[i][1] = round(prediction[i][1] * original[i][1])
        np.savetxt(resultpath+onlyfiles[i].split(".")[0]+"_gt.txt",np.reshape(prediction[i],(1,2)),fmt='%3.0f')

else:
    print("Wrong input")
