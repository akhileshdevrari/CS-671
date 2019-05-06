import numpy as np
import cv2
import tensorflow as tf
from keras.applications import VGG16
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model, load_model
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
import pickle
import os,glob
def bb_intersection_over_union(boxA, boxB):
	
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	if((xB - xA)==0 or (yB - yA)==0):
		iou = 0.0
		return iou
	interArea =  max(0, xB - xA + 1) * max(0, yB - yA + 1)                                                                                                                                                                                                                                                             
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

NEW_IMG_SIZE = (224, 224, 3)

model = load_model('model_bbox.h5')
classLB = pickle.loads(open("classbin", "rb").read())


NUM_IMG=4
paths = ['./Knuckle', './Palm', './Vein']
IoUfile = './results/IoU.txt'
with open(IoUfile,'w') as f:
    print('Cleaned file')
for path in paths:
    with open (path+'/groundtruth.txt', 'r') as fp:
        for cnt, line in enumerate(fp):
            if(NUM_IMG==cnt):
                break
            print('Line {} : {}'.format(cnt, line))
            L = line.split(',')
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

                img = image
                image = img_to_array(image)
                image /= 255
                image = np.expand_dims(image, axis=0)

                classification, reg = model.predict(image)
                clIdx = classification[0].argmax()
                clLabel = classLB.classes_[clIdx]
                (x_1,y_1,x_2,y_2) = (int(reg[0][0]),int(reg[0][1]),min(NEW_IMG_SIZE[0],int(reg[0][2])),min(NEW_IMG_SIZE[0],int(reg[0][3])))
                with open(IoUfile,'a') as f:
                    f.write(path+'/'+L[0]+" {:.2f}\n".format(bb_intersection_over_union((x_1,y_1,x_2,y_2),(x_1_,y_1_,x_2_,y_2_))*100))
                if(NUM_IMG > cnt):
                    cv2.rectangle(img, (x_1,y_1), (x_2,y_2), (255,0,0), 2)
                    cv2.rectangle(img, (x_1_,y_1_), (x_2_,y_2_), (0,255,0), 2)
                    categoryText = "{}({:.2f}%)".format(clLabel,classification[0][clIdx] * 100)
                    iouText = "IoU:{:.2f}%".format(bb_intersection_over_union((x_1,y_1,x_2,y_2),(x_1_,y_1_,x_2_,y_2_))*100)
                    cv2.putText(img, categoryText+' '+iouText, (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (6, 125, 240), 2)
                    print(categoryText+' '+iouText)
                    cv2.imwrite('./results/images/'+str(path.split('/')[1])+str(cnt)+'.png',img)
