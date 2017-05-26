# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/multiple-camera.png "Images from Multiple Cameras"
[image2]: ./img/flipped-image.png "Flipped image"
[image3]: ./img/adjust-brightness.png "Adjusting brightness"
[image4]: ./img/add_shadow.png "Add shadow"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.pdf summarizing the results
* video.mp4 a video recording of the vehicle driving autonomously around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5

```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Data Augmentation

Only sample dataset provided by Udacity is used in training. Besides images taken by the center camera, images taken by left and right cameras are also used in training. To use images of left (right) camera, steering angle is offset by +0.2 (-0.2) degree. 

![alt text][image1]

This is to simulate situations that the car is away from road center and trying to recover. Correction factor of 0.2 degree is optimum for driving speed of 9 (defined in driver.py), and a slightly larger factor (~0.25) is needed for higher driving speed.

Moreover, I use the following data augumentation technique to generate unlimited number of images for training:

* Horizontally flipping images and taking the opposite sign of the steering measurement.

* Adjusting image brightness.

* Adding shadows on the image. 

An example of image flipping:

![alt text][image2]

Examples of adjusting image brightness:

![alt text][image3]

Examples of adding random shadows:

![alt text][image4]

#### 2. Training Strategy

I split images into train and validation set in order to measure the performance at every epoch. Testing is done by running car autonomously on the simulator.

* Root-mean-squared error is used in the loss function to measure how close the model prediction to the actual steering angle.

* Adam optimizer is used for optimization with learning rate of 1e-4. 

My first step is to use a convolution neural network model similar to LeNet and I only use images from center camera for training. Trained car runs smoothly but it fails at turns because the car cannot recover from edges of the road. After adding images from left and right cameras in training, the car learns how to recover when it's away from center of road but sometimes it still fails at turns. Then I increase number of parameters in the model by making convolution layers wider and deeper. This way the model is more capable in perceiving the road. 

#### 3. Final Model 

My final model is based on the NVIDIA model, which is published in the paper "End to End Learning for Self-Driving Cars". 
This model has been proved working well on self-driving cars, so I borrow it here to fit steering angle on camera images.

* Layer 1: used to crop unnecessary part of images. Top 50 rows of pixels and bottom 20 rows of pixels are cropped away.

* Layer 2: resize images into 66x200 pixels, as is required by the NVIDIA model. 

* Layer 3: normalize images.

* Layer 4: convolution with kernel=(5,5), filter=24, strides=(2,2), valid padding and ELU as activation function.

* Layer 5: convolution with kernel=(5,5), filter=36, strides=(2,2), valid padding and ELU as activation function.

* Layer 6: convolution with kernel=(5,5), filter=48, strides=(2,2), valid padding and ELU as activation function.

* Layer 7: convolution with kernel=(3,3), filter=64, strides=(1,1), valid padding and ELU as activation function.

* Layer 8: convolution with kernel=(3,3), filter=64, strides=(1,1), valid padding and ELU as activation function.

* Layer 9: dropout to reduce overfitting in the model.

* Layer 10: flatten layer.

* Layer 11: fully-connected with 100 neurons.

* Layer 12: fully-connected with 50 neurons.

* Layer 13: fully-connected with 10 neurons.


| Layer (type)                 | Output Shape              | Param #
| -----------------------------| --------------------------| -----------
| cropping2d_1 (Cropping2D)    | (None, 90, 320, 3)        | 0
| lambda_1 (Lambda)            | (None, 66, 200, 3)        | 0
| lambda_2 (Lambda)            | (None, 66, 200, 3)        | 0
| conv2d_1 (Conv2D)            | (None, 31, 98, 24)        | 1824
| conv2d_2 (Conv2D)            | (None, 14, 47, 36)        | 21636
| conv2d_3 (Conv2D)            | (None, 5, 22, 48)          | 43248
| conv2d_4 (Conv2D)            | (None, 3, 20, 64)          | 27712
| conv2d_5 (Conv2D)            | (None, 1, 18, 64)          | 36928
| dropout_1 (Dropout)          | (None, 1, 18, 64)              | 0
| flatten_1 (Flatten)          | (None, 1152)              | 0
| dense_1 (Dense)              | (None, 100)               | 115300
| dense_2 (Dense)              | (None, 50)                | 5050
| dense_3 (Dense)              | (None, 10)                 | 510
| dense_3 (Dense)              | (None, 1)                 | 11

This model is trained for 8 epochs. In each epoch, I use 1000 batches of images with batch size = 64, so a total of 64000 randomly augmented images are used each epoch. 
 
