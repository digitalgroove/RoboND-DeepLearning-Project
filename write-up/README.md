[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)
# RoboND Term 1 Deep Learning Project, Follow-Me

**Write-up by Roberto Zegers**   
In this project I programmed a deep neural network for a drone camera to identify and track a person in simulation. 
I really enjoyed doing this project because it was an eye-opener to me when I confirmed that I can build and deploy my own artificial intelligence models.
Please write me an e-mail if you have any remarks or you want to share more information about the topic discussed.

![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/smrc-s6-rc-mimi-selfie.jpg)
**Image 1: The quadcopter following a moving target**  
(*Image Source: SMRC S6 RC Mini Kids Drone with Camera*)

___

### Table of Contents
**[Part 1: Implementing the Segmentation Network](#implementing-the-segmentation-network)**  
1.1 [Introduction](#introduction)  
1.2 [Development Process](#development-process)  
1.3 [Network Architecture](#network-architecture)  
1.4 [Hyperparameters related to training](#hyperparameters)  

**[Part 2: Training the Model](#training-the-model)**  
2.1 [Local Computer Tests](#local-computer)  
2.2 [Cloud Instance Setup](#cloud-instance)  
2.3 [Training and Testing Results](#training-and-testing-results)  
2.4 [Testing in Simulation](#testing-in-simulation)  

**[Part 3: Potential for Further Development](#potential-for-further-development)**  
3.1 [Future Model/Training Improvements](#implementation-improvements)  
3.2 [Object Detection With Custom Objects](#object-detection-with-custom-objects)  

**[Part 4: Setup Instructions](#setup-instructions)**    
4.1 [Installation Guide](#installation-guide)  
4.2 [Running the Project](#running-the-project)  

___

<a name="implementing-the-segmentation-network"/>  

## Part 1: Implementing the Segmentation Network  

<a name="introduction"/>  

### 1.1 Introduction  
The goal of this project is to find a specific person in images or video taken by a quad-copter camera in a simulated environment.
Deep learning is one methodology that has shown great success when used for the task of recognizing objects in images.
In this project I will create a deep neural network for object detection by defining the network architecture and training the network from scratch.
Particularly I will develop a fully convolutional network (FCN) to perform semantic segmentation. As a result, each pixel in the image is labeled with the class of its enclosing object or region.

![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/semantic_segmentation_title.jpeg)
**Image 2: An example of semantic segmentation**

<a name="development-process"/>  

### 1.2 Development Process  
Next you will find a brief description of the development process and building blocks involved in designing the aforementioned neural network.
As first step it was important to understand the basics of how simple neural networks work in order to not be confused with increasingly complex concepts later.
All concepts and steps learned are later required for understanding FCN's.

#### i. One Layer Neural Network

Starting point was to build a one layer **TensorFlow** network.
One layer networks, also called logistic or linear classifiers work well when the data is linearly separable.
This can be sufficient for tasks like image classification of simple images with low visual recognition complexity.
For example it work well on the MNIST database of handwritten digits because each character is isolated and on a clean background.

![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/Multinomial-Logistic-Classifier-compressor.jpg)   
**Image 3: A simple representation of a one layer neural network**

The development of a simple one layer Neural Network allowed me to understand and apply following concepts:
- **Data Normalization**: for machine learning algorithms, features (input values) must be rescaled so that they have the properties of a standard normal distribution with mean μ=0 and standard deviation σ=1.
This is specially important when different features have different scales (value ranges). If features would not be normalized, certain weights would update faster than others, this is so because feature values play a role in the update of weights.
- **One-Hot Encoding**: Normally labels indicate a category using a numerical value, but this can cause errors since the model will interpret the data to be in some kind of order (0 < 1 < 2, etc..)
To prevent this it is important to perform a “binarization” of all categories and then include them as a feature to train the model. In practical terms a column that contains categories using a numerical value (1, 2, 3, etc..) is split into multiple columns.
The category values are replaced by 1s and 0s depending on which column represents what category.

- **Setting of features and labels tensors**: When writing a TensorFlow program, the main object you manipulate and pass around is the Tensor. It is required to describe the type of input data in TensorFlow by creating placeholders. These placeholders don’t contain any data, but only specify the type and shape of the input.  
  
  ```python
  # Set the features and labels tensors
  features = tf.placeholder(tf.float32, [None, features_count])
  labels = tf.placeholder(tf.float32, [None, labels_count])
  ```
  
  The first dimension of shape is [None], since we want to stay flexible about how many images we actually provide. When specifying [None] the dimension can be of any length.
  The second dimension is the number of features, which in the case of an image is the number of pixel values per image.
  For example a RGB image that is 32 pixels wide x 32 pixels high results in a total of 32 x 32 x 3 = 3,072 values (features_count) for each image.

- **Weights and biases tensors**: Weights and biases have to have a initial value before kicking off the training, and we want our weights and biases to be initialized at a good enough starting point for the gradient descent optimization to proceed. 
  These are the key factors to consider:
  
  **Initialization of weights**  
  Weights of Neural Networks should be initialized to random numbers because the next layer gets the sum of inputs multiplied by the corresponding weight.
  If all weights would be the same value (e.g. zero or one), all units in the next layer will be the same too, no matter what was the input.
  A simple method is to draw the values randomly from a Gaussian distribution with mean 0 and standard deviation sigma.
  To initialize a tensor with a specified shape filled with random normal values we can use TensorFlow's random_normal() function:

  ```weights = tf.Variable(tf.random_normal([n_input, n_classes]))```

  A better way to initialize weights is to use truncated normal distribution.
  The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
  The benefit of using the truncated normal distribution is to prevent generating "dead or saturated neurons" when using an activation function like sigmoid (note: activation functions like Relu don’t have this problem). A saturated neuron occurs when it takes on a value that is too big or small and the neuron stops learning.
  On a sigmoid function if the weight value gets >= 2 or <=-2 the neuron will not learn. So, if you truncate your normal distribution you will not have this issue (at least from the initialization).  
  To initialize a tensor with a specified shape filled with random truncated normal values we can use TensorFlow's truncated_normal() function:  
  ```tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)```

  ```weights = tf.Variable(tf.truncated_normal([features_count, labels_count]))```  

  **Initialization of biases**  
  Because the gradients with respect to bias do not depend on the gradients of the deeper layers, there is no diminishing or explosion of gradients for the bias terms. 
  Additionally asymmetry breaking is provided by the small random numbers of the initialized weights.
  Therefore, it is possible and common to initialize the biases to be zero: 

  ```biases = tf.Variable(tf.zeros([labels_count]))```
  
- **Logits**: These are scores that represent the predictions calculated by our model. On this simple one layer neural network they are calculated using a linear model, which is basically a matrix multiplier and a bias.
  ```python
  # Linear Function WX + b
  logits = tf.matmul(tf_train_dataset, weights) + biases
  ```

- **Softmax**: It is a mathematical function that has the property of turning any input value into a value between 0.0 and 1.0. The Softmax function is used in the logistic regression model for multiclassification. This characteristic makes it perfect for turning logits (scores calculated by the model) into probabilities.  
  ```prediction = tf.nn.softmax(logits)```

- **Cross Entropy Function**:  It is that function that helps us to measure the distance (or error) between the probability vector that comes out of the classifier (that contains the probabilities of the classes) and the one-hot encoded vector that corresponds to the labels.
  By comparing the probabilities for each class outputted by the softmax function to the one-hot encoded labels for all the inputs and all the labels we can calculate the average distance (or error) of our model with respect to the training data.
  If we have a small distance to the correct class and a large distance to the incorrect class we are doing a good job in classifying every example in the training data.
  The smaller this average (also called loss) the better our model is performing on every example on our training data. Our next step then, is to minimize this value (the loss), optimizing the measure of the error (cross entropy).
  ```python
  # Cross entropy
  cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), axis=1)
  # Training loss
  loss = tf.reduce_mean(cross_entropy)
  ```
  
- **Optimizer**: It is the algorithm that helps us to update the Model parameters such as weights and bias values in order to minimize the loss function (or error function).
  For now we will rely on the gradient descent optimizer implemented by TensorFlow and treat it as a black box that we can simply use:  
  ```optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)```

- **Hyperparameter configuration**: hyperparameters represent the configurable values used when building a neural network. Different hyperparameters are one reason why networks are different from each other and why some models represent better what we want.
  Examples of these hyperparameters include learning rate, epochs, batch size and many more. Finding the right hyper-parameters is crucial to training success, but it can be hard to find.
  I describe more in detail how I selected Hyperparameters below under *"4. Hyperparameters related to training"*.

**One Layer Neural Network using TensorFlow:   
(Source: https://www.geeksforgeeks.org/softmax-regression-using-tensorflow/)**  
  
```python
# number of features 
num_features = 784
# number of target labels 
num_labels = 10
# learning rate (alpha) 
learning_rate = 0.05
# batch size 
batch_size = 128
# number of epochs 
num_steps = 5001
  
# input data 
train_dataset = mnist.train.images 
train_labels = mnist.train.labels 
test_dataset = mnist.test.images 
test_labels = mnist.test.labels 
valid_dataset = mnist.validation.images 
valid_labels = mnist.validation.labels 
  
# initialize a tensorflow graph 
graph = tf.Graph() 
  
with graph.as_default(): 
    """ 
    defining all the nodes 
    """
  
    # Inputs 
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features)) 
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels)) 
    tf_valid_dataset = tf.constant(valid_dataset) 
    tf_test_dataset = tf.constant(test_dataset) 
  
    # Variables. 
    weights = tf.Variable(tf.truncated_normal([num_features, num_labels])) 
    biases = tf.Variable(tf.zeros([num_labels])) 
  
    # Training computation (linear classifier) 
    logits = tf.matmul(tf_train_dataset, weights) + biases 
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
                        labels=tf_train_labels, logits=logits)) 
  
    # Optimizer. 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 
  
    # Predictions for the training, validation, and test data. 
    train_prediction = tf.nn.softmax(logits) 
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases) 
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
```
 
Note that a one layer model is relatively unintelligent. It has no components that would help it discern characteristics such as lines or shapes. It strictly examines the color value of each pixel, entirely independent of other pixels.
For harder images, we will need to extract sophisticated visual features using more layers of network representation to gain sufficient visual discriminative power so that the object can be successfully distinguished from those in other classes.
This is were Deep Neural Networks kick in. 

#### ii. Deep Neural Network (DNN)  

Next step is to add multiple layers to build a multilayer neural networks (a.k.a. Multi Layer Perceptron (MLP)) with TensorFlow. 
They are called Deep Neural Networks because they contain one or more hidden layers (apart from one input and one output layer).
Adding additional layers can help so that our model is complex enough to solve more difficult problems.

![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/nn-from-scratch-3-layer-network.png)  
**Image 4: A simple representation of a multi layer neural network**

The development of a Deep Neural Networks, with at least one hidden and a output (logits) layer, allowed me to understand and apply following methods:

- **Activation function**: The activation function transforms the inputs of the hidden layer. We select a non-linear activation function to let us model non-linear functions. Without an activation function we would simply have a linear regression model, which has limited capability. 
  In this implementation I use the **ReLU** activation function .
  
  ```python
  # Hidden Layer with ReLU activation function
  hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases)
  hidden_layer = tf.nn.relu(hidden_layer)
  ```

  ```output = tf.add(tf.matmul(hidden_layer, output_weights), output_biases)```
  
- **Regularization**: It is a useful technique that can help in improving the accuracy of regression models. 
It constrains/regularizes or shrinks the coefficient estimates towards zero. 
In other words, this technique discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.
  - **L2 Regularization**: a new term is added to the loss, which penalizes large weights. It adds another hyper-parameter to tune.
  - **Dropout layer**: It is another technique for regularization. A certain number of values that go from one layer to the other (activations) are set to zero completely randomly.
    This forces the network to learn redundant representations, which makes the network more robust and prevents overfitting. The dropout rate can be adjusted by setting the new hyperparameter **keep_probability**.
- **Over-Fitting**: Over-fitting can be recognised by comparing the accuracy the model achieves on training data compared with the accuracy achieved on the validation data set. If the accuracy is higher for the training set, that suggests the model has been over-fit to the training data. It's not a problem for a small amount of over-fitting.
The simplest way to prevent overfitting is to reduce the size of the model, i.e. the number of learnable parameters in the model (which is determined by the number of layers and the number of units per layer).
 
#### iii. Convolutional Neural Network (CNN)  

In CNN's weight are calculated by moving a filter or kernel (smaller than the whole size of the input) across an input to create the output.
CNNs allow the algorithms to learn patterns by taking advantage of the fact that feature (pixels in an image) are close together for a reason and that that has a special meaning.
Convolution layers reduce the size, progressively squeezing the spacial dimensions, but increase the depth of the layers. This is a fundamental characteristic of every convolutional neural networks.
To create a size reduction different techniques such as **pooling** (explained below) can be used.

![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/Conv-Neural-Network.png)  
**Image 5: A simple representation of a Convolutional Neural Network (CNN)**   
_Source: https://www.slideshare.net/perone/deep-learning-convolutional-neural-networks_

The development of a Convolutional Neural Network allowed me to understand and apply following concepts:
- **Stride**: The amount by which the filter slides horizontally or vertically to focus on a different piece of the image is referred to as the 'stride'. 
  The stride is a hyperparameter that can be tuned. Increasing the stride reduces the size of the model by reducing the number of total patches each layer observes. However, this usually comes with a reduction in accuracy.
- **Filter depth**: Is the number of neurons to which each filter or kernel connects to. The depth of the network is increased layer after layer, and will correspond roughly to the logical complexity of the representation.
  It is important to keep in mind that too few kernels could possibly lose information and overfit to specific patterns, while too many kernels could possibly underfit.
  There is no equation or exact rule of restricting the number.
- **Convolution output shape**: The formula for calculating the new height and width is:

  ```python
  new_height = (input_height - filter_height + 2 * Padding)/Stride + 1
  new_width = (input_width - filter_width + 2 * Padding)/Stride + 1
  ```

  The new depth is equal to the number of filters (see Filter depth above).

  In TensorFlow:  

  ```python
  # Apply Convolution
  conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
  # Add bias
  conv_layer = tf.nn.bias_add(conv_layer, bias)
  # Apply activation function
  conv_layer = tf.nn.relu(conv_layer)
  ```

- **Max Pooling**: A pooling layer is generally used to decrease the size of the output in the spatial dimension space (height and width). Note that the z dimension, the depth or number of layers, remains unchanged in the pooling operation. 
  A max pooling filter operates by sliding across the input layer and outputs the maximum value of the filter square. The size reduction helps to save on processing time by reducing the number of parameters to learn.
  Max Pooling also helps to prevent over-fitting by providing an abstracted form of the representation. 
  
  ![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/maxpool_animation.gif)  
  **Image 6: Animation of max pooling over a 4x4 feature map with a 2x2 filter and stride of stride of 2**  
  *Source: https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks*

- **1x1 Convolutions**: they are used to increase or decrease the number of depth channels, depending on if the number of filters of the 1x1 convolution is greater or smaller than the depth of the previous layer.
  In other words they change the dimensionality in the filter space (depth) maintaining the spatial dimension space (height and width). Often they are used to reduces the dimensions, as a means to reduce the number of computations.
  They do so by letting the network train how to reduce the dimension most efficiently.
  
  ![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/full_padding_no_strides_transposed_small.gif)  
  **Image 7: Animation of a 1x1 convolution**  
  *Source: https://iamaaditya.github.io/2016/03/one-by-one-convolution/*
  
- **Fully Connected Layer**: they are identical to the layers in a traditional deep neural network in which each neuron is connected to every neuron in the previous layer, and each connection has it's own weight.
  They act as a classifier in that each of the neurons in the Fully Connected Layer will over time capture elements of the input data that should allow it to then predict the correct value.
  In order to be able to feed the output of a convolution or pooling layer into a fully connected layer we have to flatten the array by transforming a tridimensional (W-(s-1), H - (s-1), N=depth) tensor into a monodimensional tensor (a vector) of size (W-(s-1))x(H - (s-1))xN.
  After applying the Softmax as the activation function in the output layer of a Fully Connected Layer the sum of output probabilities from the Fully Connected Layer is 1. Note there can be one or more of these layers at the end of a CNN.  
  


#### iv. Fully Convolutional Network (FCN)  
  
FCN allow to perform object detection and localization because while doing the convolution they preserve the spacial information throughout the entire network.
Structurally a FCN is comprised of two parts: Encoder and Decoder.
The Encoder Block is a series of convolutional layers that extracts features of the image.
The Decoder Block upscales the output of the Encoder such that it is the same size o the original picture.
Finally, I will combine the above two and create the model. In this step I will be able to experiment with different number of layers and filter sizes for each to build my model.

![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/convolutional_encoder-decoder.png)  
**Image 8: A simple representation of a Fully Convolutional Network (FCN)**

The development of a Convolutional Neural Network allowed me to understand and apply following techniques:
- **Conversion of fully connected to 1x1 convolutional layers**: in traditional CNN's the output of a convolutional layer is fed into a fully connected layer, and for that the array is flatten into a monodimensional tensor (a vector).
  This results in the loss of spatial information, because no information about the location (2D arrangement) of the pixels is preserved. 
  If we use a 1x1 convolutional layer instead the output tensor will preserve it dimensions and the spacial information will be preserved.
- **Upsampling**: We already saw that in Convolutional Neural Networks the intermediate layers typically get smaller and smaller (although often deeper). In order to produce full a image segmentation, that matches the width and height of the original input image we need to upsample the intermediate tensors. 
  Through the use of transposed convolutional layers (a.k.a. “deconvolutions”) we can associate a single input activation with multiple outputs.  
  In TensorFlow, the API `tf.layers.conv2d_transpose` is used to create a transposed convolutional layer.  
  ![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/full_padding_no_strides_transposed.gif)  
  **Image 9: No padding, no strides, transposed convolution**  
  _Source: https://github.com/vdumoulin/conv_arithmetic_  
  An optimized version of separable convolutions has been provided in the utils module of the provided project repo and can be implemented as follows:  
  ```
  output = BilinearUpSampling2D(row, col)(input)
  ```

- **Skip connections**: In the upsampling step the network can lose a lot of resolution. One option is to add "skip connections" from earlier higher-resolution layers. Skip connections are used to improve the coarse segmentation maps produced by FCN's because of loss of information that is the consequence of pooling. Small object detection is one example where skip connection would bring clarity in edges, texture, shapes etc of the features.
  Skip connections pass on higher resolution information from an earlier layer to a deeper layer bypassing at least one layer. Two existing methods used to implement skip connections are summation (element-wise addition), as well as a simple concatenation.
  ```
  from tensorflow.contrib.keras.python.keras import layers
  output = layers.concatenate([input_layer_1, input_layer_2]) 
  ```
- **Separable Convolutions**: Separable convolutions, also known as depthwise separable convolutions, comprise of a convolution applied separately to each channel of the input and the outputs are concatenated. Then a 1x1 convolution that takes the output channels from the previous step and then combines them into an output layer is applied.
  Depthwise separable convolutions considerably reduce the number of parameters and computation used in convolutional operations while increasing representational efficiency.
  An optimized version of separable convolutions has been provided in the utils module of the provided project repo. It is based on the tf.contrib.keras function definition and can be implemented as follows:  
  ```output = SeparableConv2DKeras(filters=32, kernel_size=3, strides=2,padding='same', activation='relu')(input)```
- **Batch Normalization**: Batch normalization is based on the idea that, instead of just normalizing the inputs to the network, we normalize the inputs to layers within the network. 
  The core observation is that this is possible because normalization is a simple differentiable operation. 
  In the implementation, applying this technique usually amounts to insert the BatchNorm layer immediately after fully connected layers (or convolutional layers), and before non-linearities. 
  In practice networks that use Batch Normalization are significantly more robust to bad initialization. Additionally, batch normalization can be interpreted as doing preprocessing at every layer of the network, but integrated into the network itself in a differentiable manner.
  In `tf.contrib.keras`, batch normalization can be implemented with the following function definition:  
  ```
  from tensorflow.contrib.keras.python.keras import layers
  output = layers.BatchNormalization()(input)
  ```

<a name="network-architecture"/>  

### 1.3 Network Architecture  

Defining the architecture of a neural network equals to setting the Hyperparameters related to the network structure:

- **Number of Layers**: As we increase the number of layers (and its size), the capacity of the neural network increases.   
  Two aspects have to be considered when determining the number of layers:  
  On the one hand, one can expect a model with higher capacity to be able to model more relationships between more variables than a model with a lower capacity.
  But on the other hand, as the model can learn to classify more complicated data, the risk to overfit the training data also grows. In other words, a model with less layers only has power to classify the data in broad strokes and could lead to better generalization.  
 
  The recommended practice is to implement model with more layers (higher capacity) and use techniques such as L2 regularization or dropout to prevent overfitting, instead of selecting a smaller capacity model (less layers).
 
- **Dropout**: unfortunately the methods provided in the utils module of the project's original Github repo do not offer a straight-forward procedure for setting the `keep_probability` hyperparameter.
  Because of this I had to select the previous smaller capacity model once I noticed that adding a new layer caused signs of overfitting. 
 
- **Convolution Kernel**: in a convolutional layer the Kernel size influences the number of parameters in a model which, in turns, influences the networks capacity.
  The *spatial dimensions of the input* in a 2D convolutional layer is determined by the height and width.
  There are no rules with regards to what height and width use for the convolution kernel. The best best value to use depends largely on the nature of the data.
  If one thinks that a big amount of pixels are necessary for the network recognize the object one will use large filters (as 11x11 or 9x9). 
  But if one thinks that what differentiates objects are some small and local features, one should use small filters (3x3 or 5x5).
  The *spatial size of the output* in a convolutional layer is determined by the stride, padding, and the depth.
  Using a stride of 1 or 2 is a widely used practice, but again, there is no hard rule about which stride to use. 
  One can set the filter depth to any value. However the more filters, the more image features can get extracted and the better the network becomes at recognizing patterns.
  It is also recommended that each successive layer has two to four times the number of filters in the previous layer. This helps the network learn hierarchical features.
  
  In this project, to change the convolution kernel hyperparameters one has to adjust the filters and kernel_size inside the `separable_conv2d_batchnorm()` and `conv2d_batchnorm()` functions:   

  ```
  def separable_conv2d_batchnorm(input_layer, filters, strides=1):
      output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides, padding='same', activation='relu')(input_layer)
    
      output_layer = layers.BatchNormalization()(output_layer) 
      return output_layer

  def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
      output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(input_layer)
    
      output_layer = layers.BatchNormalization()(output_layer) 
      return output_layer
  ```	
	
**Implementation**  
My first approach was to build the smallest possible FCN, composed of one encoder block, one 1x1 convolution and one decoder block.
I selected a small filter (3x3) since smaller filters (3x3 or 5x5) usually perform better than larger filters.
To test different architectures I progressively added more encoder/decoder layers and observed the result in accuracy.
When doing so, I found it challenging to find out, how to pass in the right arguments to the decoder block function in order for the concatenation process to work.
It was not immediately obvious to me, how to, given a "upsampled layer", find the corresponding "large input layer" to perform the decoding.
Because the concatenation process requires inputs with matching shapes I added to each tensor a code line to print its shape.  

Each time I added a layer to test a model that is one layer deeper, I doubled the number of filters in the new layer. I did so because it is a widely used practice.  

In order to be able to benchmark the different architectures I decided to fix the learning rate at an arbitrary value of 0.01.
I also fixed the number of epochs to an arbitrary value of 15 (to spare the limited computational resources available (GPU time)).

Please refer to the Performance Results section of this document for details on how well each of the tested FCN models produced predictions.  

<a name="hyperparameters"/>  

### 1.4 Hyperparameters related to training

Hyperparameters related to the network structure were already described by me above under section "1.3 Network Architecture".
In this section I will describe how I determined the hyperparameters related to training such as learning rate, batch size, etc..
Finding the right hyper-parameter values is crucial to get a good performance, next I explain how I proceeded.

First I determined what batch size value I will use on my tests:

- **Batch size**: Divides the data into batches and then uses these subsets of the dataset instead of all the data at one time for training. 
  This provides the ability to train a model, even if a computer lacks the memory to store the entire dataset.
  Larger batch size means more "accurate" gradients, in the sense of them being closer to the "true" gradient one would get from whole-batch (processing the entire training set at every step) training.
  Consequently, choosing a higher batch sizes can allow for stable training with higher learning rates.  
  But if the system has memory restrictions one has no other alternative than choosing a smaller batches to be able to run the model at all. 
  When doing so it is preferable to choose a smaller learning rate as well.
  For my particular configuration, I tried some different batch sizes and used the highest possible number that did not throw an error.  
  On my local computer, which I used to test and debug code, I did the training with a batch size of 10.
  When training the neural net on the AWS cloud instance I did the training with a batch size of 100.
  Note: I later learned that it is a widespread practice to set the batch size to a common sizes of memory (32, 64, 128, 256, etc..).
 
Next I moved on to decide on how to define what learning rates I will consider, as it is one of the most important hyper-parameters to tune:

- **Learning Rate**: It defines how quickly a network adjusts its parameters (weights) with respect to the calculated loss gradient.
  The following formula shows the relationship:   
  ```new_weight = existing_weight — learning_rate * gradient```    
  High learning rates speed up the learning but if too high, the model may not converge, or the final accuracy can be low.
  Low learning rates slow down the learning process but if too low, learning never progresses.
  In between there is a band of “just right” learning rates that successfully train.  
  
  Running the same model exploring all of the possible learning rates is impractical. So I decided to test my models accuracy at 12 different learning rates ranging from 0.000001 to 10 at logarithmic intervals:
  
  0,000001 - 0,000004 - 0,000019 - 0,000081 - 0,000351 -  0,001520  
  0,006579 - 0,028480 - 0,123285 - 0,533670 - 2,310130 - 10,000000  
  
  To run my tests I started from the lowest learning rate and increased it on each test iteration.
  
Then I had to determine the number of epochs to use:
  
- **Epoch**: An epoch is a single forward and backward pass of the entire training dataset.
  Increasing the number of epochs can increase the accuracy of the model without requiring more data.
  However this increase comes at the cost of higher training times (GPU time), which I had limited to 100 hours.
  At first, during the test runs for finding the best model architecture, I decided to fix the number of epochs to an arbitrary value of 15 (to not consume excessive GPU time).
  Later, during the test runs for finding the best learning rate I decided to let the machine run for 20 epochs.
  Once the best learning rate is found one can increase the number of epochs for that particular learning rate to improve results further.
  For instance, for the best performing learning rate found, one could set the number of epochs at a much higher value, such as 100.
  When doing so it is important to monitor the validation accuracy to see after which epoch it stops improving (to avoid wasting GPU time).

Finally these are the other training Hyperparameters that I decided not to optimize for these test runs:
  
- **Steps per epoch**: number of batches of training images that go through the network in 1 epoch.
  The number of steps times the batch size indicates the number of samples taken from the pool of training data (whole dataset). In practical terms this reduces the size of the whole dataset, thus the data processed per epoch.
  The provided default value was 200, which I did not change.
  One can calculate a tailor-made value for steps_per_epoch based on the total number of images in training dataset divided by the batch_size.

- **Validation steps**: number of batches of validation images that go through the network in 1 epoch.
  This is similar to steps_per_epoch, except validation_steps is for the validation dataset.
  I used the default value (50) for this as well.

<a name="training-the-model"/>  

## Part 2: Training the Model

<a name="local-computer"/>  

### 2.1 Local Computer Tests

My Local Computer was the starting point to build and debug the neural network.
Working on my local PC had the advantage of not consuming cloud GPU time resources.
Specs: Intel Core i7-4500U CPU (2 CPU Cores, 1.80 GHz, 2.60 GHz Turbo), 8GB RAM, Windows 10.
	
To start the notebook server from the command line first run:

`$ source activate RoboND`

This will activate the conda environment which will provide access to the required libraries. Then run:

`$ jupyter notebook`

The Notebook Dashboard should open in a new browser tab.

**Training example**  

A 3 encoder/decoder model (learning rate 0.01, batch size 10, 2 epochs) took roughly 1 hour time.  
Training results: loss = 0.0533, val_loss = 0.0714, final grade score = 0.003640328513150

Predictions compared to the mask images:
![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/images/model_3b_2e_local_patrol_without_target.png)  
![](https://github.com/digitalgroove/RoboND-DeepLearning-Project/blob/master/write-up/images/model_3b_2e_local_following_target.png)  

Once I defined and tested the different neural network architectures I switched on to the AWS cloud instance to train them.

<a name="cloud-instance"/>  

### 2.2 Cloud Instance Setup

<a name="training-and-testing-results"/>  

### 2.3  Training and Testing Results

<a name="testing-in-simulation"/>  

### 2.4 Testing in Simulation

<a name="potential-for-further-development"/>  

## Part 3: Potential for Further Development 

<a name="implementation-improvements"/>  

### 3.1 Future Model/Training Improvements

<a name="object-detection-with-custom-objects"/>  

### 3.2 Object Detection With Custom Objects  

<a name="setup-instructions"/>  

## Part 4: Setup Instructions

<a name="installation-guide"/>  

### 4.1 Installation Guide

<a name="running-the-project"/>  

### 4.2 Running the Project

___

**References:**
- https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
- http://wiki.fast.ai/index.php/Over-fitting
- https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
- https://www.mabl.com/blog/image-classification-with-tensorflow
- https://iamaaditya.github.io/2016/03/one-by-one-convolution/
- http://cs231n.github.io/neural-networks-1/
- https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
