
---
layout: post
title:  "Assignment-1"
date:   2019-03-15 18:47:43 +0530
categories: jekyll update
---

Detailed problem statement and our code/submissions is available [here](https://github.com/akhileshdevrari/CS-671/tree/master/Assignment-1).

# Task 1: Data Management - Line Dataset Generation
Create a numpy zeroes array of size 28x28. Then starting from a random (row, col) draw a line of length {7, 15} and width {1, 3}. Fill this line with color {(255,0,0), (0,0,255)}. Now rotate this line with Image.rotate() function of PIL to required angles and save the images in corresponding class folders using Image.save().

[problem-statement-assgn1]: https://github.com/SerChirag/CS-671-1

# Task 3: Implementing Layers API using numpy

Given two image data-sets, we were required to use code a Layer API to build a fully-connected neural network, in order to classify the images.

We were successfully able to code the required API by using Python Library Numpy. No other curated neural networks library like TensorFlow, Keras or PyTorch were used.
We constructed two major classes :

 1. Layer Class : This class is used to design individual layers of the neural network. Using this class the user can decide the activation functions and the dropout probability.
 2. Model Class : This class is used to tune the entire network as a whole. Using this class, the user can decide the Optimizer, the learning rate, the number of maximum iterations.

The various specifications provided in the Layers API :

 1. Activation Functions : Sigmoid, Rectified Linear Unit, tanh, Softmax.
 2. Regularization : Inverted Dropout.
 3. Optimizer : Gradient Descent, Gradient Descent with Momentum, RMS Prop, Adagrad and Adam. 
 4. Loss Function : Cross-entropy.

[More Information](https://github.com/akhileshdevrari/CS-671/blob/master/Assignment-1/Task-3/Neural_Network_Layer_API.pdf)
