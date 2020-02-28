# Assignment 2

## Table of Contents
- [Part 1: Pre-processing](#part-1-pre-processing)
- [Part 2: Hyperparameters](#part-2-hyperparameters)
- [Part 3: Homework Tutorial](#part-3-homework-tutorial)

## Part 1: Pre-processing
I’d like to emphasize a fundamental concept in data science: pre-processing.

Machine learning is not magic, as most of you already know. Your neural network will perform not as well as it should if you mindlessly throw your training data into it (it will probably perform quite poorly). Many of you may have heard of a phrase similar to “feed a neural network garbage, and you will get a garbage neural network.” A lot of what makes a good neural network is not the neural network architecture itself -- it’s the data that you use to train it. In simpler terms, we need to make our data as “intuitive” as possible for the machine to learn by removing things that are unnecessary to a prediction. This entails determining what part of the data you would consider important, and what part of the data is not.

Let’s take a very simple example: I would like to create a machine learning algorithm to detect the leafy part of a strawberry. Let's assume that the images I will test/validate on *only* has strawberries.

![strawberry leaf](https://i.imgur.com/WnsH1fm.jpg)

Now I would be able to just throw this exact image above into a model, telling the model “this is the leafy part of a strawberry." However, I can do a simple threshold of the image to only keep certain colors within the image, and get something like this:

![thresholded strawberry leaf](https://i.imgur.com/rb9n4fM.png)

As you can see, I’ve thresholded colors in a way that everything except the greens of the image are simply turned black. Let’s assume that through the feature extraction, what the neural network ends up “seeing” is the edges of the picture:

Unprocessed | Processed
------------ | -------------
![unprocessed edge](https://i.imgur.com/O16cN9k.png) | ![processed edge](https://i.imgur.com/0zOprgZ.png)

Unprocessed, there seems to be a lot of edges that are not even part of the leafy part of the strawberry. But on the right, we can see that most of that noise is gone. Just by thresholding certain colors, we were able to get an image without unnecessary data (such as edges that aren’t part of the leaves of a strawberry).

Removing unnecessary data can be very beneficial for how your model performs. This example is very simple, but I’d like to emphasize this so you may keep this in mind whenever you’re training your models.

## Part 2: Hyperparameters
You may have noticed word “parameters” occasionally been thrown left and right in class (don’t quote me on that). While internal parameters are parameters that are set during training (such that their values change as the neural network trains), hyperparameters are parameters that are set before training.

Some common hyperparameters you may encounter during coding:
* Learning rate
* Number of neurons per layer
* Filter size (convolutions)
* Activation function per layer
* Number of convolutions
* Dimensionality of data

Why is this important? Depending on the hyperparameters you choose before even training your neural network may drastically improve (or degrade) the performance of your neural network. That being said, you should pick hyperparameters that make sense for the goal that you’re trying to achieve. This should go without saying, but sometimes this is not something completely intuitive to think about.

For example, a look at the sigmoid activation function below. It is apparent that the steepness of the curve between -2 and 2 are much steeper relative to the steepness from -6 to -4, and 4 to 6.

![sigmoid](https://i.imgur.com/lN4ZskZ.png)

When using this in binary classification, where you’re trying to output to -1 or 1, this is great! This looks like a smooth step-function, and when data is passed through this activation function, data in the middle will end up having steep differences in value compared to the extremities, in this case closer to 1 or 0. This makes the data less ambiguous when categorizing, for example.

You can kind of visualize what a sigmoid does to data through these images**:

![sig-images](https://i.imgur.com/IKoJG8I.png)


<sub> ** Note: Not really accurate, but is used as a visual analogy. Image from http://ccis2k.org/iajit/PDF/vol.1,no.2/10-nagla.pdf </sub>

You can think of it as the pixels that are already very dark *don't* change value a lot, but the pixels within the image that are roughly halfway in the middle of black and white in the original image get pushed closer to the extremities (in this case, they turn whiter).

Using sigmoids is good in this scenario, but you do not always want to do this. If you’re not predicting binary classifications but rather regressions, using sigmoids doesn't make a lot of sense.

Take my one of my research projects that focuses on human motion data. Let's assume that in the GIFs you see below, the pixels from the left to right represent some x-values. We are then taking the x-values of my foot and putting it through a "sigmoid":

"Linear" Activation | "Sigmoid" Activation
------------ | -------------
![norm](https://i.imgur.com/HTLcaSJ.gif) | ![sig](https://i.imgur.com/6TWh0QF.gif)

<sub> Note: Do not @ me about my legs or leg day </sub>

Again, this is not really accurate to what would actually happen to the data, but hopefully this is a good enough example as to why you wouldn't want to use something like a sigmoid for this type of data. In the "sigmoid" GIF, you may notice how my foot motion ends up on either the very left or the very right most of the time, but motion in the middle is almost non-existent, making the movement unnatural for walking. If your goal is to generate natural walking data, then applying this kind of activation function to your data is like giving your neural network bad information to learn with, and you'll find that the performance will be quite horrible. Think of it as trying to write Chinese when all you've been given to learn from is the alphabetical system.

## Part 3: Homework Tutorial

Okay, let's do some coding! We're going to be making an MLP neural network in Keras, and doing hyperparameter grid search using Talos. Additionally, we are going to be doing some very rudimentary pre-processing to our data today.

### Required library
Before we start, please install talos. This is the library that we will be using in order to do hyperparameter grid search. It has a lot of features, such as plotting of neural network performance for each change of a hyperparamter. More information can be found here: [Talos](https://github.com/autonomio/talos)

Using Anaconda Prompt (Windows), or terminal (macOS or Linux), activate your python environment and type this to install this library to your python interpreter. If you're lost, you may refer to the machine setup guide where you do roughly the same thing.

```
pip install talos
```

### Defining the neural network

For people who have experience coding using Keras, this may be a little different. We'll be wrapping our model into a function because of how Talos does its hyperparameter grid search. As you may know, this is not necessary normally (though may be recommended). The code may also be a little different from what you may already be doing.

#### Imports

Let's import what we need

```python
import talos
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend
```

#### Dataset

For this tutorial, we'll be using MNIST for our dataset. It is a dataset that contains 28 x 28 px images of the digits 0 to 9.

![mnist](md_res/mnist.png)

Thus, let's go get the data. Note how all of the values within images are being divided by 255. As many of you would know, the most common image formats store pixel values from a range of 0 to 255. Dividing this by 255 will allow the image ranges to be 0 - 1, which can be make-or-break some activation functions.

<sub>(Note that the following code is taken from a Keras example)</sub>
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

```

To create the label data, they make a binary classification matrix:
```python
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
```

What does the above do? Essentially, this changes how the label data is formatted. Why is this important? Lets say that the digit in question is a 6:

![mnist_6](md_res/6.png)

A ```to_categorical``` representation of a label (within your target/y/"answer" array) attached to this would be:

```[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]```

where at index 6 is a ```1```, while all other indices are a ```0```.

Note that, when using to_categorical, **the amount of nodes/dimensionality of your output layer must equal the length of the label array**. As such, the above label is of length 10, and as such we need 10 output nodes (for this homework).

Why would we use this? This is actually really helpful rather than using scalar variables like what we did in the last assignment. Not only does it increase accuracy, but given certain activation functions for your output layer, can allow for *probabilities* for the output.

Lets say that a trained model is predicting a 6. The output may look something like this:
[0.15, 0, 0, 0.10, 0, 0.10, 0.55, 0, 0.10, 0]

The prediction would still be 6, as it has a probability of 55%, but we can also see the probability of predictions that it is making. This is something not possible with scalar representations for labels.

#### Building the neural network

##### Making a model

Now that we have our data, let's build a neural network using Keras. We're going to be making a serial model today, and the easiest way to do that in Keras is using ```Sequential()```.

First, instantiate the model. We can call the Sequential that we imported:

```python
model = Sequential()
```

##### Adding layers

Using Sequential, you can easily add layers to the model by calling ```model.add()```. Some examples of layers are
* Dense (the simplest type of layer, essentially a neuron with a weight, bias, and activation attached)
* LSTM (classified as a recurrent neural layer with memory for temporal data)
* Conv2D (classified as a convolutional neural layer, in which CNNs excel in spatial data)
* MaxPooling (commonly used in CNNs for dimensionality reduction and translational invariance purposes)

In this tutorial, we're going to be making an MLP neural network, the simplest kind. Thus, we will be adding a Dense layers in our neural network.

Before we do that, let's add a ```Flatten()``` layer, as Dense can only take vectors/one-dimensional arrays. Right now, through the code we used to fetch and process the MNIST dataset, each image in the dataset is actually a 3D array of this format: (28, 28, 1), where
* 28, - represents the row of pixels
* 28, - represents the column of pixels
* 1 - represents the single integer that denotes the luminosity of the pixel (0 for black, 255 for white, and gray in between)

The flatten layer just flattens this 3-dimensional array representing an image into a single dimension or a vector, So instead of a 28 x 28 x 1 array, you have an array of length 768, with each element in that array being the luminosity of the pixel. The reason why we're doing this is because Dense layers can only take a vector or single dimension array as input.

```python
model.add(Flatten(input_shape=(28, 28, 1)))
```
Inside our ```Flatten()```, we add ```input_shape```. ```input_shape``` is a parameter you need to set in the first layer that you ```add()```. It should represent the shape that the training data has.

Other layers found in Keras, such as ```Conv2D```, can take a 3D array (2D for the row x column, 1D for the values). I will explain that more in-depth in the next homework.

Now, let's add the Dense layers. Note that for every ```add()``` in a sequential model represents another layer in the neural network, with the last layer added being the output layer, and the first layer added being the input layer.

```python
model.add(Dense(units=24, activation='softmax'))
model.add(Dense(units=10, activation='softmax'))
```
Where:
* Units represents the amount of neurons per layer
* Activation represents the activation function used in that layer

**Keep in mind that the last layer you add is your output layer. Therefore, it should have the same number of neurons to the output shape that you need.** For example, we are training on digits from 0-9, giving us 10 different categories. This is the reason why the last layer we add has 10 neurons/units.

##### Compiling and fitting your model

After adding all of your desired layers, let's compile the model using ```model.compile()```.
```python
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
```
Where:
* Loss represents the loss used by the model
* Optimizer represents the optimizer used by the model
* Metrics specifies the metrics

Lastly, let's fit our training and testing data into the model we compiled using ```model.fit()```.
```python
out = model.fit(x=x_train, y=y_train, batch_size=2000, epochs=100, verbose=0)
```
Where:
* X represents training data
* y represents your label
* Batch size represents the size of your mini-batch
* Epochs represents the number of iterations you are doing for training

We've built the neural network! Your code finished should look something like this:
```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(units=24, activation='softmax'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, batch_size=2000, epochs=100)
```
Running this code should entail your neural network training on the MNIST dataset.

### Doing hyperparameter grid search
We've discussed a few hyperparamters in the previous section, namely:
* Number of units
* Activation function
* Loss
* Optimizer
* Batch Size
* Number of epochs

And for each, there are many choices. Let's say you don't have any intuition for what activation function between softmax, sigmoid, and ReLU will perform best on your dataset. What you can do is a hyperparameter grid search. This is an automated way of trying every combination of loss, optimizer, # units, etc. that you specify.

Lets instantiate a dictionary *param_dict* that contains the hyperparameters that we want to be testing:
```python
param_dict = {
    'units': [12, 24],
    'activation': ['softmax', 'sigmoid'],
    'loss': ['mse', 'binary_crossentropy'],
    'optimizer': ['adam', 'adagrad'],
    'batch_size': [1000, 2000]
}
```

As you can see, we set units number, activation, loss, and optimizer params. This gives us 2 x 2 x 2 x 2 x 2 = 32 different combinations of these hyperparamters.

Now, let's wrap our model in a function. We'll be implementing these explicit parameters, and be returning both the model and our fit function (stored in 'out' variable), both variables that Talos needs.

```python
def my_model(x_train, y_train, x_val, y_val, params):
```

We can paste the code from what we have earlier inside this function, however, we'll be making some changes. We'll be changing the hyperparameters from before to be able to intake parameters passed in by Talos. Thus, we'll be using the index of our params implicit parameter in order to set the hyperparameters. Here is what your model inside a function should look like:

```python
def my_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))

    model.add(Dense(units=params['units']))
    model.add(Dense(units=10, activation=params['activation']))
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    out = model.fit(x_train, y_train,
                    validation_data=[x_val, y_val],
                    batch_size=params['batch_size'],
                    epochs=20,
                    verbose=0)
    return out, model
```
As you can see, we are changing these hyperparameters in the model into ones that call from param_dict dictionary:
* Units of all layers but the last
* Activation
* Loss
* Optimizer
* Batch size

Now that we have the model wrapped in a function, we can call the Talos function to run the hyperparameter grid search, namely Scan(). Note that Scan() will not print anything when it is done. Instead, it will generate a .csv file within directory as determined by the ```experiment_name``` parameter.

```python
talos.Scan(x_train, y_train, p, my_model, x_val=x_test, y_val=y_test, experiment_name="talos_output")
```

Finally, your hyperparameter grid search code should look like this:
```python
import talos
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend

p = {
    'units': [12, 24],
    'activation': ['softmax', 'sigmoid'],
    'loss': ['mse', 'binary_crossentropy'],
    'optimizer': ['adam', 'adagrad'],
    'batch_size': [1000, 2000]
}


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def my_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))

    model.add(Dense(units=params['units']))
    model.add(Dense(units=10, activation=params['activation']))
    model.compile(loss=params['loss'],
                  optimizer=params['optimizer'],
                  metrics=['accuracy'])

    out = model.fit(x_train, y_train,
                    validation_data=[x_val, y_val],
                    batch_size=params['batch_size'],
                    epochs=20,
                    verbose=0)
    return out, model

talos.Scan(x_train, y_train, p, my_model, x_val=x_test, y_val=y_test, experiment_name="talos_output")
```

Go ahead and run that code to train it! **Note: if you want to see the per-epoch updates change verbose in ```model.fit()``` from 0 to 1**
