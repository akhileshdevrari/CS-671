Extract the files in the same directory which contains "Data" and "Mask", lets call it root.


################################################################################################

Make sure you have following files/folders in root:

1. An empty folder named "results". This will contain all the graphs produced after successful execution of code. After is time you want to rerun the code, empty the "results" folder.

3. A files named "results.out" this will contain output data like, accuracy and loss.

################################################################################################



To run the code, just type the following command on terminal"

python3 main.py




################################################################################################

Please make sure following libraries are already installed:

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.utils import shuffle
from keras import optimizers

import sys



################################################################################################

After successful execution, following results will be produced:

File named "results.out" will be filled with output data

Folder named "results" will be filled with output plots



################################################################################################

Please alter the number of epochs on line number 228