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

#### 2. Training Strategy

My GPU is a low-end GeForce GTX 745 (4GB memory) whose processing power (gflops) is only 1/10 of GTX1080. Training large models on this GPU is really time consuming, so I need to use relatively simple models in training. My first step is to use a convolution neural network model similar to LeNet and I only use images from center camera for training. Trained car runs smoothly but it fails at turns because the car cannot recover from edges of the road. After adding images from left and right cameras in training, the car learns how to recover when it's away from center of road but sometimes it still fails at turns. Then I increase number of parameters in the model by making convolution layers wider and deeper. This way the model is more capable in perceiving the road. 

My final model contains 3 convolutional layers, and it is trained for 8 epochs. After 8 epochs, training loss keeps decreasing but validation loss starts increasing slowly, meaning overfitting of the model. I try to add dropout in each convolutional layer and train the model for 20 epochs. Validation loss decreases monotonically in these 20 epochs but its magnitude is higher than the model without dropout. Also performance of the car is worse than before. So in the end I use the model without dropout and only train 8 epochs.

#### 3. Final Model 

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

* The first layer (cropping2d) is used to crop unnecessary part of images. Top 50 rows of pixels and bottom 20 rows of pixels are cropped away.

* Layer lambda_1 is to resize images into 64x64 pixels, since high resolution images are not necessary for driving on this track. This way we reduce degree of freedom and save a lot of training time. 

* Layer lambda_2 is to normalize images.

* There are 3 convolutional layers followed by 3 fully connected layers. RELU is used as activation function in all layers.

* The model uses an adam optimizer, so the learning rate was not tuned manually (model.py line 110). 

* After training, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 4. Next Step

I hope the car can run on the second track even if it's only trained on images from the first track, however the car runs out of road right after start. Difference in the second track is that there are many shadows on the road and the brightness of camera images changes from time to time. To let the model learn about shadows and change of brightness, dataset is augmented using the following functions:

~~~~
def adjust_brightness(image):
    # make image darker
    return np.array(image*np.random.uniform(0.5, 1.0), dtype=np.uint8)

def add_shadow(image):
    # add a random dark block in image to simulate shadow
    height, width = image.shape[:2]
    corner_y = np.random.randint(height-20)
    corner_x = np.random.randint(30,width)
    image[corner_y:,:corner_x,:] = image[corner_y:,:corner_x,:]*0.3
    return image
~~~~

I have tried deeper models with up to 6 convolutional layers on these augmented dataset, however the car still fails to run on the second track. Need to explore more on data augmentation.



