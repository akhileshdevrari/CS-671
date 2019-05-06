import numpy as np
import cv2
import os, glob
import random
import tensorflow as tf

NEW_IMG_SIZE = (224,224, 3)
name=[]
bbox=[]
NUM_IMG=40
path = "./Four_Slap_Fingerprint/Ground_truth"

for cnt,filename in enumerate(glob.glob(path+'/*.txt')):
    # if(NUM_IMG==cnt):
    #     break
    imagename = "./Four_Slap_Fingerprint/Image/"+filename.split('/')[3].split('.')[0]+'.jpg'

    # print(filename, imagename)
    image = cv2.imread(imagename)
    # print(image.shape)
    if image is not None:
        name.append(imagename)
        IMG_SIZE = image.shape
        # img = image
        image = cv2.resize(image,(NEW_IMG_SIZE[1], NEW_IMG_SIZE[0]))
        # print(img.shape,image.shape)
        with open(filename,'r') as fp:
            lbox=[]
            for i,line in enumerate(fp):
                L=line.split(',')
                x_1 = int(L[0])
                y_1 = int(L[1])
                x_2 = int(L[2])
                y_2 = int(L[3][:-1])
                x_1_ = int(np.round(NEW_IMG_SIZE[1]*(x_1/IMG_SIZE[1])))
                y_1_ = int(np.round(NEW_IMG_SIZE[0]*(y_1/IMG_SIZE[0])))
                x_2_ = int(np.round(NEW_IMG_SIZE[1]*(x_2/IMG_SIZE[1])))
                y_2_ = int(np.round(NEW_IMG_SIZE[0]*(y_2/IMG_SIZE[0])))

                box = [x_1_, y_1_, x_2_, y_2_]
                # lbox.append(x_1_)
                # lbox.append(y_1_)
                # lbox.append(x_2_)
                # lbox.append(y_2_)
                lbox.append(box)
                # break
                # cv2.rectangle(image, (y_1_,x_1_), (y_2_,x_2_), (0,255,0), 2)
                # cv2.rectangle(img, (y_1,x_1), (y_2,x_2), (0,255,0), 2)
        # cv2.imwrite('./Take2/2/'+'finger.png',image)
        # cv2.imwrite('./Take2/2/'+'finger_original.png',img)
        bbox.append(lbox)
bbox=np.array(bbox)
name=np.array(name)
print(bbox.shape,name.shape,bbox[0])

np.save('fbbox.npy',bbox)
np.save('fname.npy',name)