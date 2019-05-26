# Task-3 : Core Point Detection in Fingerprints

## Problem Statement
Core point refers to the centre point of a fingerprint image and is found in the middle of spiral. Task is to find ”Core Point” on a fingerprint image captured through different sensors.

## Pre-processing

 1. **Converted position of Core-Point as a ratio of the image dimensions.** This helped in dealing with images of various dimensions. 
 Initially, the regression problem was giving results that were not even in the image dimension. For example, in image of 320*480, the neural network detected the core-point at (2580,1368), which is impossible. On doing this pre-processing, the core-point detected was atleast within image dimension.
 
 2. **Converted all images to dimensions to 320*480 pixels using linear interpolation.**
 3. **Converted all images to black-and-white using Adaptive Threshold** : This made sure that only data of finger-print was recorded, and the background gradient noise was filtered out. 
 

>     im = cv2.imread(input_data,cv2.IMREAD_GRAYSCALE)
>     tho = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5)

 4. **Orientation Field using Sobel filters** According [research,](https://www.sciencedirect.com/science/article/pii/S1110866513000030) the core-point is located in a vortex with ridges forming a unique orientation pattern around the core-point. 
 

 

>     def go(img):
>         sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
>         sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
>         po = 2 * sobelx * sobely
>         go = pow((sobely*sobelx),2)
>         a = 0.5*np.arctan2(sobely,sobelx)
>         oy = cv2.blur(np.sin(a),(5,5))
>         ox = cv2.blur(np.cos(a),(5,5))
>         angle = sigmoid(np.arctan2(oy,ox))
>         return angle

![enter image description here](https://lh3.googleusercontent.com/MxDoaggHCjco6TTMWgYX-TAzg9-IT3mK7d3-5O-HL91LrQNKUlbYG8N7a0BLunAiECNul3O260_c "Sample Pre-processing")

## Model

I have used a simpler version of AlexNet

