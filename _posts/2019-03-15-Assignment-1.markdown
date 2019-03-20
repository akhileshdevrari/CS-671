---
layout: post
title:  "Assignment-1"
date:   2019-03-15 18:47:43 +0530
categories: jekyll update
---
[github repo]: https://github.com/akhileshdevrari/CS-671/tree/master/Assignment-1/
[task1]: https://github.com/akhileshdevrari/CS-671/tree/master/Assignment-1/Task-1
[task2]: https://github.com/akhileshdevrari/CS-671/tree/master/Assignment-1/Task-2
[task3]: https://github.com/akhileshdevrari/CS-671/tree/master/Assignment-1/Task-3
[task3 report]: https://github.com/akhileshdevrari/CS-671/blob/master/Assignment-1/Task-3/Neural_Network_Layer_API.pdf

Detailed problem statement and our code/submissions is available [here][github repo].

# Task 1: Data Management - Line Dataset Generation

>codes are available [here][task1]

## Approach/Methodology

Create a numpy zeroes array of size 28x28. Then starting from a random (row, col) draw a line of length {7, 15} and width {1, 3}. Fill this line with color {(255,0,0), (0,0,255)}. Now rotate this line with Image.rotate() function of PIL to required angles and save the images in corresponding class folders using Image.save().

# Task 2: Computational Graph - Gravity Simulator

>codes are available [here][task2]

We were given masses, positions and velocities of 100 particles in 2-D coordinate system. Only gravitational force was acting on them. We have to calculate final positions and velocities of each particle when minimum distance between any two particles was below some threshold (0.1 units in this case). Time interval for each iteration was 0.0001 sec.

We coded the solution using different approaches.

1. Using Tensorflow library with a for/while loop in Session.run().
2. Using Tensorflow library with [tf.while_loop()](https://www.tensorflow.org/api_docs/python/tf/while_loop) in Session.run().
3. Normal python code using numpy array.

## Subtasks

### 1.  Create computational graph using Tensorboard

>![tb_img](tb.png)

### 2.  Compare performance of code written using tensorflow with normal python code

>Normal python code took 28 sec while computational model tensorflow code took only 2 sec to run. Both codes ran for 330 iterations until termination.

# Task 3 : Layer API - A Simple Neural network

>codes are available [here][task3]

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

[Detailed Report](https://github.com/akhileshdevrari/CS-671/blob/master/Assignment-1/Task-3/Neural_Network_Layer_API.pdf)
