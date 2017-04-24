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
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

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

Moreover, dataset is augmented by flipping images and taking the opposite sign of the steering measurement. For example:

![alt text][image2]

A total number of 48216 images were used in training.

#### 2. Model 

| Layer (type)                 | Output Shape              | Param #
| -----------------------------| --------------------------| -----------
| cropping2d_1 (Cropping2D)    | (None, 90, 320, 3)        | 0
| lambda_1 (Lambda)            | (None, 64, 64, 3)         | 0
| lambda_2 (Lambda)            | (None, 64, 64, 3)         | 0
| conv2d_1 (Conv2D)            | (None, 60, 60, 32)        | 2432
| max_pooling2d_1 (MaxPooling2 | (None, 30, 30, 32)        | 0
| conv2d_2 (Conv2D)            | (None, 26, 26, 32)        | 25632
| max_pooling2d_2 (MaxPooling2 | (None, 13, 13, 32)        | 0
| conv2d_3 (Conv2D)            | (None, 9, 9, 64)          | 51264
| max_pooling2d_3 (MaxPooling2 | (None, 4, 4, 64)          | 0
| flatten_1 (Flatten)          | (None, 1024)              | 0
| dense_1 (Dense)              | (None, 512)               | 524800
| dense_2 (Dense)              | (None, 64)                | 32832
| dense_3 (Dense)              | (None, 1)                 | 65

* The first layer (cropping2d) is used to crop unnecessary part of images.

* Layer lambda_1 is to resize images into 64x64 pixels, since high resolution images are not necessary for driving on this track. This way we reduce degree of freedom and save a lot of training time. 

* Layer lambda_2 is to normalize images.

* There are 3 convolutional layers followed by 3 fully connected layers. 

* The model uses an adam optimizer, so the learning rate was not tuned manually (model.py line 110). 

* After training, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to LeNet. I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
