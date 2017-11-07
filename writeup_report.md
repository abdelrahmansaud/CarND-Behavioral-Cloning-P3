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

[image1]: examples/center.png "Center driving"
[image2]: ./examples/recovering_1.png "Recovery Image"
[image3]: ./examples/recovering_2.png "Recovery Image"
[image4]: ./examples/before_flip.png "Before flipping"
[image5]: ./examples/after_flip.png "After flipping"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 58-63) 

The model includes RELU layers to introduce nonlinearity (code line lines 58-63), and the data is normalized in the model using a Keras lambda layer (code line 54). 

This is followed by 4 fully connected layers (number of units 100, 50, 10, 1) that makes the angle prediction

#### 2. Attempts to reduce overfitting in the model

Overfitting was controlled by the number of epochs, in this case 1 epoch through the augmented dataset was sufficient. (model.py lines 72). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 72). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and augmented the images as described in the below section.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a model that can extract features from images, and use them to predict the steering angle.

My first step was to use a basic convolution neural network model that consists of convolutional neural network layers that act as feature extractors followed by fully connected layers to do the regression and predict the steering angle. I thought this model might be appropriate because this architecture is well proven by many well known models such as VGG16, VGG-19, LeNet, in which they differ in terms of hyperparemeters (such as, hyperparameters, kernel size, activation function, etc) so a basic model that followed the same idea was used as a start.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I used number of epochs hyperparameter to control it, and since overfitting seemed to occur during the first epochs (less than 10) there was no need to use Keras Callbacks

Then I added a cropping layer to the model to remove the lower part of the image is just showing the car's hood and providing no useful information to the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 52-69) consisted of a convolution neural network similair to what NVIDIA uses for their self-driving car [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) that has 10 layers, a normalization layer, cropping layer, 5 convolutional layers, and 3 fully connected layers.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the road. These images show what a recovery looks like starting from side of the track :

![alt text][image2]
![alt text][image3]

I also used images from side cameras as points for non-centered driving, and by adjusting the steering angle it provided another example for the model on getting back to center

Then I recorder another two laps driving in opposite direction, because the track has turn bias to one direction, those two laps were used to balance that.

To augment the data sat, I also flipped images and angles thinking that this would further combat that bias in direction of turns the track has. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1. I used an adam optimizer so that manually training the learning rate wasn't necessary.
