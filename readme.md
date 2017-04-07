==Traffic Sign Classifier Project

The steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results


[//]: = (Image References)

[image1]: images/example.png "Example of augmented data"
[image4]: images/1.jpg "Traffic Sign 1"
[image5]: images/11.jpg "Traffic Sign 2"
[image6]: images/14.jpg "Traffic Sign 3"
[image7]: images/18.jpg "Traffic Sign 4"
[image8]: images/40.jpg "Traffic Sign 5"

---
=== Writeup / README

Link to my [project code](https://github.com/paghdv/TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

=== Data Set Summary & Exploration

The code for this step is contained in the second code cell of the IPython notebook.  

I used python to calculate summary statistics of the traffic signs data set:

Before augmenting and balancing the sizes are as following

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 42

====2. The training data was augmented and balanced

I decided to augment data by rotating it from -20 to +20 degrees in 5 degrees intervals. After the data was augmented I balanced the training set and kept 278392 images equality distributed for all classes. 

===Design and Test a Model Architecture

====1. Data pre-processing

After the training data was augmented and balanced I proceeded to normalize the images by substracting them 128 and dividing them by 128 in order to keep their intensity values between [-0.5,+0.5]
Data normalization equalizes the impact of the intensities of the images during training. The data was then shuffled in order to avoid skewing the learning towards a certain class right from the begining.
![Example of augmented data][image1]


====3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 13th cell of the ipython notebook. It is based on LeNet model with a few modifications:
1. Changed the input size to acept rgb images
2. Modified fully connected layers size to be able to encode more classes
3. Added dropout regularizers in the fully connected layers

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| Input 400, Output 120							|
| RELU					| 												|
| Dropout				| (Used 0.5 while training)						|
| Fully connected		| Input 120, Output 84							|
| RELU					| 												|
| Dropout				| (Used 0.5 while training)						|
| Fully connected		| Input 84, Output n_classes					|
|:---------------------:|:---------------------------------------------:|
 


====4. Training

In the cell 19 I define my training which uses an adam optimizer. After a few tests I used a training rate of 0.0001, with 30 epochs and batch sizes of 2048

====5. Results

My final model results were:
* training set accuracy of 97.9%
* validation set accuracy of 93.7% 
* test set accuracy of 92.3%

If an iterative approach was chosen:
* The first step towards choosing an architecture was to start from a known and simple one: LeNet-5. This model was chosen in order to see siginificant improvement while modifying it. Other options like GogleNet could possibly perform better in this dataset but the objective was not necesarily to obtain the best scores.
* The initial architecture had a slightly stronger over-fitting and didn't have enough room to encode the number of the classes in the output (under-fitting). Therefore the main changes were to add a dropout as a regularizer and added more nodes.
* Which parameters were tuned? How were they adjusted and why? 

===Testing the model on New Images

Here are five German traffic signs that I found on the web:

![30km/h][image4] ![Priority road][image5] ![Stop sign][image6] 
![General caution][image7] ![Roundabout mandatory][image8]

==== Acuracy of the model in new images

All five images were well classified with the resulting net. This is slightly surprising but at the same time an example of five images is not necessarely statistically significant. The test and validation accuracy suggest than more that at least 4 of those images could be classified correctly.

==== Prediction certainty of the model
The code for making predictions on my final model is located in the last cell of the Ipython notebook.

For the first image, the model is quite sure that this is a 30km/h sign (probability of 0.9999), which is absolutely correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999        			| 30km/h 	  									| 
| 1e-6     				| Beware of ice/snow							|
| 1e-8					| Dangerous courve to the left					|
| 1e-9	      			| Pedestrians					 				|
| 1e-12				   	| Slippery Road      							|


For the other images the predictions were in the same level of certainty, whith the correct class predicted with confidences of at least 0.999.
