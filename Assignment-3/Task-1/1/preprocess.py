import numpy as np
import cv2
import os

import tensorflow as tf

NEW_IMG_SIZE = (224, 224, 3)

name=[]
label=[]
bbox=[]
# NUM_IMG=2

f1= open("./Knuckle/groundtruth.txt","r")
f2= open("./Palm/groundtruth.txt","r")
f3= open("./Vein/groundtruth.txt","r")
f = np.array(f1.readlines()+f2.readlines()+f3.readlines())
np.random.shuffle(f)
paths = ['./Knuckle', './Palm', './Vein']
for cnt,line in enumerate(f):
            # if(NUM_IMG==cnt):
            #     break
            # print('Line {} : {}'.format(cnt, line))
            L = line.split(',')
            path=""
            if(L[5][:-1] == 'knuckle'):
                path="./Knuckle"
            if(L[5][:-1] == 'Palm'):
                path="./Palm"
            if(L[5][:-1] == 'veins'):
                path="./Vein"
            image = cv2.imread(path+'/'+L[0])
            if image is not None:
                IMG_SIZE = image.shape
                image = cv2.resize(image,(NEW_IMG_SIZE[1], NEW_IMG_SIZE[0]))
                x_1 = int(L[1])
                y_1 = int(L[2])
                x_2 = int(L[3])
                y_2 = int(L[4])

                x_1_ = int(np.round(NEW_IMG_SIZE[1]*(x_1/IMG_SIZE[1])))
                y_1_ = int(np.round(NEW_IMG_SIZE[0]*(y_1/IMG_SIZE[0])))
                x_2_ = int(np.round(NEW_IMG_SIZE[1]*(x_2/IMG_SIZE[1])))
                y_2_ = int(np.round(NEW_IMG_SIZE[0]*(y_2/IMG_SIZE[0])))
               
                name.append(path+'/'+ L[0])
                label.append(L[5][:-1])
                box = [x_1_, y_1_, x_2_, y_2_]
                # boxA = [x_1, y_1, x_2, y_2]
                # boxC=[i/224 for i in box]
                # boxD=[x_1/IMG_SIZE[1], y_1/IMG_SIZE[0], x_2/IMG_SIZE[1], y_2/IMG_SIZE[0]]
                bbox.append(box)
name=np.array(name)
label=np.array(label)
bbox=np.array(bbox)
np.save('label.npy', label)
np.save('name.npy', name)
np.save('bbox.npy', bbox)
print(name.shape,label.shape, bbox.shape)
print(label[:10],name[:10],bbox[:10])