# Recurrent Neural Networks
## Short Reading
### Overview
Recurrent Neural Networks (RNN) differ from CNNs by taking into consideration previous inputs for the current prediction. If in a CNN all previous input is discarded, in an RNN previous inputs are retained in some fashion to help what's being predicted. Because of this trait, they generally perform well on data with temporal features, and generally perform much worse than CNNs for data with only spatial features (e.g. image recognition).

Thus, when it comes to sequences, the first thing that people usually try to employ is a recurrent neural networks, such as an LSTM.
Remember the waveform I used in the CNN tutorial to make a point between 1D and 2D Convolutions?
![singlewaveform](https://i.imgur.com/m9mVQSs.png)

With this type of data, if you're trying to learn the actual waveform, it may be better to use an RNN. This is the case, for example, if you're trying to predict the next value within a time series, such as:
![tsprediction](https://i.imgur.com/1QTZnXV.png)

<sub> image taken w/o permission from https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/</sub>

### Applications of RNNs
To help tie this into real world applications, there are papers that use some CNN in order to learn features of the current frame of a video, and use an RNN in order to learn the features from one video frame to the other (essentially the correlations of the previous frames to the current frame).

There are many applications of RNNs today, many of them in natural language processing (NLP). You have LSTM autoencoders that can handle sequence to sequence predictions (much like in the last homework where we were doing image to image predictions), and are sometimes used for language translation (sentence to sentence), language models or generative models to generate new words given context, etc.

There is a pretty interesting generative model that is RNN based as well, from Google's DeepMind team: PixelRNN and PixelCNN.

![pixelrnncompletion](https://i.imgur.com/9DDBNVS.png)

The Pixel models work by generating values *pixel by pixel* rather than a computation all at once. Thus, for image completion for example, it is important for the model to know what the previously generated pixels were before computing the next.

However, just because you have sequences does not mean you have to always use an RNN. You have to take into consideration how big of a time horizon you need to take into consideration. For example, if you're going to only predict over a very short time horizon, CNNs may still be used. If you're going to predict over a long time horizon, where long-term dependency is pretty important, then RNNs are usually a better idea.

An example of this is text or word classification. Yes, a word is a sequence of letters, but just like how an image is a sequence of pixels, you are not trying to learn the temporal relations of each letter, but rather how the entire word looks ("at once"). Thus, there are a lot of models for word classification that are based on CNNs. RNNs, on the other hand, can be seen when whole sentences need to be generated, or translated. This is because each word's relation to each other has to be learnt. Thus, if your model needs to generate the next word in a sentence given the previous words, RNNs are the way to go.

### Backpropagation Through Time (BPTT)
As what I believe what covered in class, because RNNs don't only consider current outputs but also previous outputs, backpropagation is also a little different in RNNs. The main difference between backpropagation in feed-forward neural networks and RNNs is that at each time step, the gradient weight W are summed up.

### LSTMs and GRUs
Even then simple RNNs, such as:

![simplernnunit](https://i.imgur.com/AXlVa2q.png)

<sub> image taken w/o permission from http://colah.github.io/posts/2015-08-Understanding-LSTMs/ </sub>

...aren't very good with data that have long-term dependencies, as it suffers from a vanishing or exploding gradient problem during BPTT. The vanishing or exploding gradient problem can be simply explained by certain inputs + the memory that a neuron have would to continue to increase (explode) or continue to decrease (vanish) to the point in which the neuron is either too strong within the prediction (exploding gradient problem) or too weak (vanishing gradient problem). In fact, this is the reason why variations such as Long-Short Term Memory (LSTM) or Gated Recurrent Units (GRU) exist.

In contrast, this is what an LSTM unit looks like:

![lstmunit](https://i.imgur.com/XPHFHe1.png)

<sub> image taken w/o permission from http://colah.github.io/posts/2015-08-Understanding-LSTMs/ </sub>

And this is what a GRU unit (a variation on the LSTM unit) looks like

![gruunit](https://i.imgur.com/ySa2X9N.png)

<sub> image taken w/o permission from http://colah.github.io/posts/2015-08-Understanding-LSTMs/ </sub>

LSTMs were created in order to fix this long-term dependency problem, by fixing the vainishing/exploding gradient problem. They do this by allowing the modal neuron to have both a memory gate as well as a forget gate rather than a single layer. Because of their multi-layer repeating module, LSTMs are able to remember a lot more information than their simpler RNN counterparts (with a singular repeating module).

As seen from the above network, input going through an LSTM unit undergo a multistep process:
1. First, a forget gate, that determines whether or not it wants to remember or forget some input on a scale of 0 to 1.
2. Second, we decide which information to remember by the evaluation of candidate values
3. Third, we update the old cell state with the new one.
4. Lastly, we figure out what to output and output it.

This is a pretty heavy summarization of what goes on in an LSTM layer, a slightly more indepth explanation with equations and all can be found here (which is also my images sources): http://colah.github.io/posts/2015-08-Understanding-LSTMs/


### LSTM Deeper Dive
If you'd like to know more about RNNs and LSTM (specifically for MLP) this is a good, multipart resource into these models:

http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/ (this is part 1 of the series)

## Letter Sequence Tutorial
This tutorial partially handed off to another tutorial by Jason Brownlee (many of you may have seen his tutorial before):
https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

Unlike the homework you'll be tackling for submission (in which I'll leave the original author to explain the code) I'll be explaining this code here as there are a lot of things that I think he has left out, that is important to understanding how this code works.

The objective with the code I've taken is to generate the next letter in the alphabet, given the current letter. Not only that, but the output of the next letter is fed again as input and the model has to predict the letter after that, etc.

This sequence can be visualized as
1. Start at A: Model prediction is B.
2. Take previous prediction B: model predicts C.
etc.

However, if there is a mistake, such as:
1. Start at A: Model prediction is C.
2. Take previous prediction C: model predicts D
etc.
Then obviously the model did not learn the sequence correctly.

From the above, here is our problem that we are posing for the neural network: the neural network needs to learn the inter-dependency of each letter.

Here is what you'll need to import:
```py
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
```

```py
# fix random seed for reproducibility
numpy.random.seed(7)
```

Here, we are going to define the alphabet as one long string that we iterate through. In addition, we're doing the usual character to int (which you can do ord as well) as you've already learnt, NNs train on numbers.

```py
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
```

We're defining a sequence length of 1 as we're going to learn each letter *one by one* instead of learning, for example, 3 letters to predict the next one. In addition, you can see that for every character in your training array (dataX), the "label" character in the test array (dataY) is the next character in the sequence.

```py
seq_length = 1
dataX = []
dataY = []

for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)
```

The important thing to note is that LSTM will take a 3-dimensional input, wherein one of the dimensions will take the amount of time steps within the data. As you can see by the next line, dataX is being reshaped to

* The number of data samples there are (26)
* The number of time steps there are (1, sequence length, as we're learning one character at a time
    * This can be increased if you're trying to learn over multiple time steps. If you're trying to learn over 3 time steps, raise this to 3.
* The number of features each data sample has (1, as a letter only has one scalar value to represent it)
    * This can be increased if your data sample has multiple data points, such as if your data has height and width (2 features) over time.

```py
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))
```

Now we're going to do the usual normalization, so that values are between 0 and 1. In addition, we're going to do the usual one-hot encoding that I've talked about before. Take note that dataX and dataY have now been set to new variables X & y.

```py
X = X / float(len(alphabet))
y = np_utils.to_categorical(dataY)
```

Again, batch size is equals to 1 as we don't want to generalize an output over multiple inputs, just 1. This is important to note as batch size plays an important role in how statefulness works in Keras.

The rest of the layers are probably pretty intuitive at this point: LSTM layer to extract temporal features within the data, and have a Dense layer that outputs the "classification." You can think of this classification as classifying the input into one out of 26 categories.

However, let's focus on the LSTM layer
The LSTM's hyperparameters are defined here as
* Number of nodes
* Batch input shape (which is just (1, 1, 1))
* Statefulness

Statefulness is not something that we've cared about in Keras before, as statefulness does not matter in non-RNNs. When a model is not stateful (stateful=False, which is default in Keras) in every sequence the cell states within the LSTM is *reset*. This means whatever state the LSTM achieved, it will not be propagated in the calculation of the next batch. Simply, all cell states in the node are reset together after *each and every batch calculation*. On the other hand, if a model is stateful (stateful=True), then whatever cell state is attained from computing at at index i will be used in the calculation of i + batch size.

This is why batch size is important in statefulness. It is because for whatever batch size that you end up setting, it will have to calculate not only what index to calculate next, but also compute the dimensionality of the states that need to be propagated for the calculation of the *next batch*. Because it is carried as input to the next calculation, its dimensionality has to be defined as it is considered as part of the input shape. The structure that stores these states are of shape (batch size, output dimensionality).

As we've said already, we're keeping it at 1 as we only want to learn a letter at a time.

```py
batch_size = 1
model = Sequential()
model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Another big difference compared to previous neural networks is how the model is trained. As you can see, we're not doing a model.fit() with epochs=300, but instead a model.fit() inside a for loop with epochs=1. This is because we want to manually do something in between fit(), which is to reset_states().

Before I explain the reset_states(), I want to talk about shuffle=False as it ties into the statefulness explained above.
We don't want to shuffle samples in X. Why? Simply because our X is already structured sequentially from what is the truth. We want the model, between each batch, to predict the letter at index i + 1 from the letter at index i. To summarize, the correlations between samples in X with index i and index i + 1 are lost when shuffling of the samples happens.

Now you may think, "isn't model.reset_states() in the below code exactly what you've explained as stateful=False?" Not quite: Unlike stateful=False that resets cell states *after every batch*, model.reset_states() in our code below resets states *after every epoch of training*, as seen by the manual input of the line in the for loop.

You can think of it as if we do the LSTM(stateful=False), the states will be reset 25 times per epoch in the model.fit() below. However, because we're doing LSTM(stateful=True), the states will be reset *manually* through the line model.reset_states(). This ensures that the model learns the correlations *between each letter* in each epoch, but when it comes to relearning the entire alphabet starting from A all the way to Z, it attains some fresh, new cell states for optimization. Not completely accurate, but if it helps you remember, we're resettings states as we *don't* want to keep the correlations from sequence to sequence, but rather the correlations from character to character.

```py
for i in range(300):
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()

scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
model.reset_states()

print("Model Accuracy: %.2f%%" % (scores[1]*100))
```

Proof of model learning the alphabet

```py
seed = [char_to_int[alphabet[0]]]
for i in range(0, len(alphabet)-1):
    x = numpy.reshape(seed, (1, len(seed), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    print(int_to_char[seed[0]], "->", int_to_char[index])
    seed = [index]
model.reset_states()
```

Proof of model statefulness

```py
letter = "K"
seed = [char_to_int[letter]]
print("New start: ", letter)
for i in range(0, 5):
    x = numpy.reshape(seed, (1, len(seed), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    print(int_to_char[seed[0]], "->", int_to_char[index])
    seed = [index]
model.reset_states()
```

And here is your whole code, which is almost exactly the same as what's found in the source:
```py
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

# fix random seed for reproducibility
numpy.random.seed(7)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

seq_length = 1
dataX = []
dataY = []

for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)


X = numpy.reshape(dataX, (len(dataX), seq_length, 1))

X = X / float(len(alphabet))
y = np_utils.to_categorical(dataY)

batch_size = 1
model = Sequential()
model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
for i in range(300):
    model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()

scores = model.evaluate(X, y, batch_size=batch_size, verbose=0)
model.reset_states()

print("Model Accuracy: %.2f%%" % (scores[1]*100))


seed = [char_to_int[alphabet[0]]]
for i in range(0, len(alphabet)-1):
    x = numpy.reshape(seed, (1, len(seed), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    print(int_to_char[seed[0]], "->", int_to_char[index])
    seed = [index]
model.reset_states()


letter = "K"
seed = [char_to_int[letter]]
print("New start: ", letter)
for i in range(0, 5):
    x = numpy.reshape(seed, (1, len(seed), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    print(int_to_char[seed[0]], "->", int_to_char[index])
    seed = [index]
model.reset_states()
```

## Text Generator Tutorial
This tutorial is handed off to another tutorial by Trung Tran. The objective is to generate text given a dataset, i.e.:
![resultoftg](https://i.imgur.com/n1UhVVX.png)

This was my result training on Alice in Wonderland after 13 Epochs. And as most of you may know, this *is* a sign of overfitting, as it is most likely memorizing words and regenerating them. However, through this tutorial I hope you can see that these types of things are possible.

Code source and intuitional explanations are found at this person's GitHub. It is pretty well explained:
https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/

The code had some missing lines (lines he did not include for working code) and errors, written with the previous TensorFlow version, and was quite hard to read, so I've rewritten the code and will be walking you through it.

As usual, we will be importing the following libraries. Note that we are importing the Long-Short Term Memory (LSTM) layer, as well as the TimeDistributed layer that allows us to apply a specific layer to every time step within the LSTM unrolling step. It is also here where we define some variables and hyperparameters not only for our neural network training but also for shaping our training/target datasets.
```py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed, Activation, Dense
import string

DATA_DIR = 'alice.txt'
SEQ_LENGTH = 100
HIDDEN_DIM = 700
LAYER_NUM = 3
BATCH_SIZE = 12
```

Next, we'll be sifting the data being read. I'm using text from *Alice in Wonderland*, which may be found [here](https://www.gutenberg.org/files/11/11.txt).
```py
data = open(DATA_DIR, 'r').read()

valid_characters = string.ascii_letters + ".,! -'" + string.digits
character_to_int = {}
int_to_character = {}
for index in range(len(valid_characters)):
    character = valid_characters[index]
    character_to_int[character] = index
    int_to_character[index] = character

training_string = ""
for character in data:
    if character in valid_characters:
        training_string += character
    elif character == '\n':
        training_string += ' '

while True:
    if "  " in training_string:
        training_string = training_string.replace("  ", ' ')
    else:
        break

target_string = training_string[1:] + training_string[0]

X = []
y = []
for i in range(0, len(training_string), SEQ_LENGTH):
    training_sequence = training_string[i:(i + SEQ_LENGTH)]
    integer_training_sequence = [character_to_int[value] for value in training_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, len(valid_characters)))
    if len(integer_training_sequence) == SEQ_LENGTH:
        for j in range(SEQ_LENGTH):
            input_sequence[j][integer_training_sequence[j]] = 1.
    X.append(input_sequence)

    y_sequence = target_string[i:(i + SEQ_LENGTH)]
    print(training_sequence, '|', y_sequence)
    y_sequence_ix = [character_to_int[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, len(valid_characters)))
    if len(y_sequence_ix) == SEQ_LENGTH:
        for j in range(SEQ_LENGTH):
            target_sequence[j][y_sequence_ix[j]] = 1.
    y.append(target_sequence)

X = np.reshape(X, (-1, SEQ_LENGTH, len(valid_characters)))
y = np.reshape(y, (-1, SEQ_LENGTH, len(valid_characters)))

model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, len(valid_characters)), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(len(valid_characters))))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam")


def generate_text(model, length):
    ix = [np.random.randint(len(valid_characters))]
    y_char = [int_to_character[ix[-1]]]
    X = np.zeros((1, length, len(valid_characters)))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(int_to_character[ix[-1]], end="")
        ix = np.argmax(model.predict(np.array(X[:, :i + 1, :]))[0], 1)
        y_char.append(int_to_character[ix[-1]])
    return ''.join(y_char)

GENERATE_LENGTH = 20
nb_epoch = 0
while True:
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH)
    if nb_epoch % 10 == 0:
        model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))
```

```py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed, Activation, Dense
import string

DATA_DIR = 'alice.txt'
SEQ_LENGTH = 100
HIDDEN_DIM = 700
LAYER_NUM = 3
BATCH_SIZE = 12

data = open(DATA_DIR, 'r').read()

valid_characters = string.ascii_letters + ".,! -'" + string.digits
character_to_int = {}
int_to_character = {}
for index in range(len(valid_characters)):
    character = valid_characters[index]
    character_to_int[character] = index
    int_to_character[index] = character

training_string = ""
for character in data:
    if character in valid_characters:
        training_string += character
    elif character == '\n':
        training_string += ' '

while True:
    if "  " in training_string:
        training_string = training_string.replace("  ", ' ')
    else:
        break

target_string = training_string[1:] + training_string[0]

X = []
y = []
for i in range(0, len(training_string), SEQ_LENGTH):
    training_sequence = training_string[i:(i + SEQ_LENGTH)]
    integer_training_sequence = [character_to_int[value] for value in training_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, len(valid_characters)))
    if len(integer_training_sequence) == SEQ_LENGTH:
        for j in range(SEQ_LENGTH):
            input_sequence[j][integer_training_sequence[j]] = 1.
    X.append(input_sequence)

    y_sequence = target_string[i:(i + SEQ_LENGTH)]
    print(training_sequence, '|', y_sequence)
    y_sequence_ix = [character_to_int[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, len(valid_characters)))
    if len(y_sequence_ix) == SEQ_LENGTH:
        for j in range(SEQ_LENGTH):
            target_sequence[j][y_sequence_ix[j]] = 1.
    y.append(target_sequence)

X = np.reshape(X, (-1, SEQ_LENGTH, len(valid_characters)))
y = np.reshape(y, (-1, SEQ_LENGTH, len(valid_characters)))

model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, len(valid_characters)), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(len(valid_characters))))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam")


def generate_text(model, length):
    ix = [np.random.randint(len(valid_characters))]
    y_char = [int_to_character[ix[-1]]]
    X = np.zeros((1, length, len(valid_characters)))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(int_to_character[ix[-1]], end="")
        ix = np.argmax(model.predict(np.array(X[:, :i + 1, :]))[0], 1)
        y_char.append(int_to_character[ix[-1]])
    return ''.join(y_char)

GENERATE_LENGTH = 20
nb_epoch = 0
while True:
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH)
    if nb_epoch % 10 == 0:
        model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))
```

## Sources
1. https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/
