---
layout: post
title:  "Assignment-2"
date:   2019-04-10 20:47:43 +0530
categories: jekyll update
---
[github repo]: https://github.com/akhileshdevrari/CS-671/tree/master/Assignment-2/
[task1]: https://github.com/akhileshdevrari/CS-671/tree/master/Assignment-2/Task-1
[task2]: https://github.com/akhileshdevrari/CS-671/tree/master/Assignment-2/Task-2
[task3]: https://github.com/akhileshdevrari/CS-671/tree/master/Assignment-2/Task-3
[task3 report]: https://github.com/akhileshdevrari/CS-671/blob/master/Assignment-2/CS_671_2_Report.pdf

Detailed problem statement and our code/submissions is available [here][github repo].

# Task 1: Foundations of Convolutional Neural Networks

>codes are available [here][task1]

Create a sequential model using keras api with tensorflow backend for mnist(10 classes) and line dataset(96 classes). Tweak the hyperparameters of the model.

# Task 2: Multi-Head Classification

>codes are available [here][task2]

We used keras api with tensorflow backend to design non-sequential model for multi-head classification of the line dataset. We designed feature network first and on top of that built 4 classification heads based on 4 different variations(length, width, color and angle).

For aggregated metrices, we assigned different weights to different classification heads and add them together for getting total metrices of the multi-head network.

![Model Graph]({{site.baseurl}}/img/multi_head_model.png)


# Task 3 : Network Visualization

>codes are available [here][task3]

We used keras model api to get details of the intermediate layers of the network for differernt test images of both mnist and line dataset and plotted them.

>![Sample intermediate layer mnist]({{site.baseurl}}/Assignment-2/Task-3/mnist/conv2d_1_grid_1.jpg)
>![Sample intermediate layer line]({{site.baseurl}}/Assignment-2/Task-3/line/conv2d_1_grid_22222.jpg)



For visualizing convnet filters, we started from a blank image and maximised the response of a particluar filter by using gradient descent technique.

>![Sample convnet filter mnist]({{site.baseurl}}/Assignment-2/Task-3/mnist/results_1.jpg)
>![Sample convnet filter line]({{site.baseurl}}/Assignment-2/Task-3/line/results_22222.jpg)

we plotted heatmap and superimposed image (heatmap+test_image) of class activations.

>![Sample test image]({{site.baseurl}}/Assignment-2/Task-3/mnist/h_test_1.jpg)
>![Sample heatmap]({{site.baseurl}}/Assignment-2/Task-3/mnist/h_heatmap_1.jpg)
>![Sample superimposed]({{site.baseurl}}/Assignment-2/Task-3/mnist/h_superimposed_1.jpg)