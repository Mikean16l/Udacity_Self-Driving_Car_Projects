# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visual.png "Visualization"
[image2]: ./examples/normal.png "normalization"
[image3]: ./examples/architecture.png "Architecture"
[image4]: ./signs/1.jpg "Traffic Sign 1"
[image5]: ./signs/2.jpg "Traffic Sign 2"
[image6]: ./signs/3.jpg "Traffic Sign 3"
[image7]: ./signs/4.jpg "Traffic Sign 4"
[image8]: ./signs/5.jpg "Traffic Sign 5"
[image9]: ./examples/new.png "predicted"
[image10]: ./examples/perform.png "perform"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Writeup / README includes all the rubric points

You're reading it! and here is a link to my [project code](https://github.com/Mikean16l/CarND-Traffic-Sign-Classifier-Project-LQT/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is 10 random pics from training set with its class number.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the images so that the data has mean zero and equal variance. By doing (data/255 - 0.5) 

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]

As a last step, I convert the images to grayscale because I want to speed up the training process, and further more the color is unlikely to give any performance but the shape of signs are more important


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

![alt text][image3]



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the follow global parameters:

* Number of epochs = 15

* Batch size =128

* Learning rate = 0.001

* Dropout = 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 0.99747

* validation set accuracy of 0.945 

* test set accuracy of 0.931


This solution based on modified LeNet-5 architecture. With the original LeNet-5 architecture, I've got a validation set accuracy of about 0.87.
 
Architecture adjustments:

* perform preprocessing (grayscale and normalization). Results for training and validation sets were 0.99 and 0.86 that's overfitting.

* Add a dropout layer with keep_prob 0.5. Results for training and validation sets are 0.99796 and 0.955


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (32x32x3):

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The sign #1 might be difficult to classify because it has noisy background.

The sign #3 might be difficult to classify because it's rotated.

The sign #4 might be difficult to classify because  it's dark.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image9]


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of 93.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

The top five soft max probabilities are:

![alt text][image10]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


