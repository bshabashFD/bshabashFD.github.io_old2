---
layout: post
title: Employing Recurrent Neural Networks for Text Generation
mathjax: true
image: writing-logo.png
---

Recurrent Neural Networks (RNNs) are naturally designed to process sequences and extract patterns from them. Unlike Feed-Forward NNs which accept simply a collection of inputs as features, RNNs can pay attention to the order of the inputs coming in, and as such are often used to explore patterns through time or through any naturally occurring sequence.

In this post, we'll explore how to employ RNNs for text processing and generation, and we'll be using [TensorFlow](https://www.tensorflow.org/install) and its [Keras](https://keras.io/) API. This post is loosely based on the tutorial at [https://www.tensorflow.org/beta/tutorials/text/text_generation](https://www.tensorflow.org/beta/tutorials/text/text_generation)

## Phase 1: Obtain the data

Lucky for us, Keras comes shipped with an example text data. Let's import everything we need, and grab our text data:

```python
import tensorflow as tf

import numpy as np
from collections import deque

from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import seed
seed(1)


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
```

Great, let's see what we have there:
```python
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print('length of text: {} chars'.format(len(text)))
vocab = sorted(set(text))
print('{} unique chars'.format(len(vocab)))

print(text[:400])
```
```
length of text: 1115394 chars
65 unique chars
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it 
```
Ahhh, that's nice, some works of William Shakespeare. I'm not sure if it's all of them (he did write a lot), but we have The Tragedy of Coriolanus, and The Tempest, to name a few, in there.

Now that we have the data, what can we do with it?

## Phase 2: Define our problem

We're going to use this text collection to train an RNN which can generate Shakespeare-like writing. We will give it a string of length $n$ characters, and it will generate the next character for us. We see we have 65 unique characters, so this will be a 65 class classification problem. We will provide a single string of length $n$, and will process it to create an output of 1 character.

Simply speaking, we'd like our network to learn to print "o" when we provide the string "hell" (to make "hello") or "e" when we provide the string "goodby" (to make "goodbye").

let's define our string length to be 100

```python
seq_length = 100
```

## Phase 3: Pre-process our data

RNNs, and really all machine learning methods, can only accept numbers as inputs, so we need first to number'ify our text. What this means is that we will assign every character to a number, and also have a way of converting those numbers back to characters

```python
# dictionary comprehension, assign every character to a number from 0-64
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# let’s check our text after it’s been transformed into numbers
text_as_int = np.array([char2idx[c] for c in text])
print(text_as_int)
```

```
[18 47 56 ... 45 8 0] 
```

Now we have a long array with numbers in it, we want to create a dataset which has a 100 number long sequence as the input (X), and a single number as the output (Y). We’re going to use a python `deque` for that, which is like a list but has limited capacity, so that when it fills up, and we enter another element, the oldest element is kicked out.

```python

X_data = []
y_data = []

newline_idx = char2idx["\n"]

# Initialize a data deque with only a newline in it, that takes care of the fact the first line of the 
# text isn't preceeded by a newline, but every other new line in the text is

data_deque = deque([newline_idx],maxlen=seq_length)


for i, char_id in enumerate(text_as_int[:-1]):
    data_deque.append(char_id)
    
    if (len(data_deque) == seq_length):
        X_data.append(list(data_deque))
        y_data.append(text_as_int[i+1])
    
    if ((i % 100) == 0):
        print(i, end="\r")

print(i)

X_data_np = np.array(X_data)
y_data_np = np.array(y_data)

print(X_data_np.shape)
print(y_data_np.shape)
```

Let’s take a look at our created datasets:
```python
# Let's take a look at the X and Y data
for i in range(5):
    print("X:\t",repr(''.join(idx2char[X_data[i]])))
    print("y:\t",repr(idx2char[y_data[i]]))
    print('-------------')
```
```
X: '\nFirst Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYo' 
y: 'u' 
------------- 
X: 'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou' 
y: ' ' 
------------- 
X: 'irst Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou ' 
y: 'a' 
------------- 
X: 'rst Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou a' 
y: 'r' 
------------- 
X: 'st Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou ar' 
y: 'e' 
------------- 
```

Looks good, the last thing we’re going to do, is shuffle the data around. We’re going to shuffle the data again later on, so let’s make a function to do that for us

```python
def shuffle_data(X_data, y_data):
    
    y_data = y_data.reshape((y_data.shape[0], 1))
    combined_data = np.hstack((X_data, y_data))
    
    np.random.shuffle(combined_data)

    X_data = combined_data[:, :-1]
    y_data = combined_data[:, -1]
    
    return X_data, y_data


X_data_np, y_data_np = shuffle_data(X_data_np, y_data_np)
print(X_data_np.shape)
print(y_data_np.shape)
```

## Phase 4: Build and Compile our Model

Finally we can create our model and train it. We’re going to use Long-Short Term Memory units ([LSTM units](https://en.wikipedia.org/wiki/Long_short-term_memory)) as our recurrent units, but you can experiment with Gated Recurrent Units (GRUs) as well.

But before we can build our model, we need to consider one more thing. Recall we just assigned a number between 0 and 64 to each of our 65 characters. This was an easy way to turn those text characters into numbers, but we ended up imposing an artificial order on them. We could figure out some representation which is more meaningful, but we can also have Keras do it for us by introducing an [Embedding Unit](https://keras.io/layers/embeddings/). In essence, an embedding unit of $n$ dimensions, takes all our alphabet (in this case 65 characters) and learns a unique vector representation for each character in this $n$-dimensional space. In our case, we will take our characters and convert each into a 25 dimensional vector.

Let’s build our Recurrent Neural Network:
```python
model = Sequential()

model.add(Embedding(len(vocab), 25, input_length=seq_length))
model.add(LSTM(1024))
model.add(Dense(len(vocab), activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam')

model.summary()
```
```
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 100, 25)           1625      
_________________________________________________________________
lstm_3 (LSTM)                (None, 1024)              4300800   
_________________________________________________________________
dense_3 (Dense)              (None, 65)                66625     
=================================================================
Total params: 4,369,050
Trainable params: 4,369,050
Non-trainable params: 0
_________________________________________________________________
```
Let’s review what we have there:

First, an embedding input layer, it takes a 100 character long sequence and converts each character into a 25 dimensional vector. Thus the output of this layer is a 100$\times$25 two dimensional matrix. 

Then, we have a 1024 neuron LSTM layer, it takes the 100$\times$25 matrix and produces a 1024 single set of values as output. Each LSTM neuron reads the 100$\times$25 matrix, learns to make sense of the sequence of 100 25-dimensional vectors, and outputs a single value, creating a total of 1024 values being passed on to the final layer. 

The final layer is a simple layer of 65 neurons, who each take the 1024-long output of the 2nd layer, and outputs a number from 0.0-1.0. Each of the 65 final neurons correspond to one of the 65 characters, so each $i^{th}$ neuron outputs the probability the output character for the input sequence is character $i$ (i.e. neuron 0 outputs the probability the next character in the sequence is character 0, neuron 1 outputs the probability the next character in the sequence is character 1, etc’).

Once the model is declared, we compile it to bring it to life, and we output the summary to get an idea of our model’s architecture.

## Phase 5: Training and Testing

We will train our RNN for 5 epochs, which means it will look at the data 5 times, but to save our poor RAM memory from exploding, we will do so in batches of 480. This means we will divide our dataset into chunks of 480 data points, and show each chunk to the model. Once all chunks have been shown, we will call it an epoch. We’ll then repeat the process 4 more times.

However, after each epoch, we’d like to see what the model has learned, so let’s define a function that will take our model, an input set of characters, and produce an output of a given size:

```python
def get_text_from_model(model, test_input, output_size=150):
    
    combined_text = test_input+""
    
    for i in range(output_size):
        if (len(test_input) > seq_length):
            padded_test_input = test_input[-seq_length:]
        else:
            padded_test_input = ("\n" * (seq_length - len(test_input))) + test_input
        text_as_int = np.array([char2idx[c] for c in padded_test_input])
        text_as_int = text_as_int.reshape((1, seq_length))

        y_predict_proba = model.predict(text_as_int)[0]
        y_id = np.random.choice(list(range(len(vocab))), size=1, p=y_predict_proba)[0]

        y_char = idx2char[y_id]
        test_input = test_input+y_char

    return test_input
```

This function basically takes our model, a seed input, and a length, and produces an output of that length with our seed input as the initial set of characters. You can see later how this works.

Let’s train our model!

```python
EPOCHS = 5       # NNs operate in epochs, meaning this is how many times the neural network will go 
                 # through the entire data
BATCH_SIZE = 480   # at each epoch, it will split the data into units of 480 samples, and train on those


for i in range(1, EPOCHS+1):
  print("EPOCH: ",i)
  X_data_np, y_data_np = shuffle_data(X_data_np, y_data_np)
  model.fit(X_data_np, y_data_np,
            batch_size=BATCH_SIZE,
            epochs=1)

  test_input = "ROMEO:"
  print(get_text_from_model(model, test_input, output_size=150))
  print("------------")
```

```
EPOCH:  1
1115294/1115294 [==============================] - 1607s 1ms/sample - loss: 1.9394
ROMEO:
Glatue to formo, leave out his brote;
upen on our fows unkinged woald them guxs:
To seak him frush no let dive; if I reved,
And then shaked these apa
------------
EPOCH:  2
1115294/1115294 [==============================] - 1607s 1ms/sample - loss: 1.3957
ROMEO:
As I beseece you, or fault! when I
I envy behold their hopys: Lord and Warwick.

GLOUCESTER:
From your brother lave him now, my his lady!

ROMEO:
Sti
------------
EPOCH:  3
1115294/1115294 [==============================] - 1603s 1ms/sample - loss: 1.2605
ROMEO:
What else?

CAMILLO:
Negely Aniinally, sir; for I mis-again!
Above the woman's head, dissoluted.

AUTOLYCUS:
Where is thou didst give madifest neiver
------------
EPOCH:  4
1115294/1115294 [==============================] - 1604s 1ms/sample - loss: 1.1743
ROMEO:
Fhonesty to guess? I have not what you were,
A fair suit doth call them by Mowbray at the house
And right in grief discharge the fay.

MERCUTIO:
hang
------------
EPOCH:  5
1115294/1115294 [==============================] - 1602s 1ms/sample - loss: 1.1038
ROMEO:
Camillo, and your arms is lie again:
Good Cape to me--I warn't it ten.
Poor horte, and you conjused your honour,
Before I shall dispose you. True and
------------
```

By the end of the first epoch, our model learned the syntax for Shakespearean writing, but it still produces some garbage language with only short words resembling English (even Shakespearean English). However, by the end of epoch 5, our model produces mostly coherent words. 
Furthermore, since we parsed our text as a set of characters, we can even give our model names and words which do not appear in the text, and it can handle them.

```python
print(get_text_from_model(model, "BORIS THE BLADE:", output_size=450))
```
```
BORIS THE BLADE:
Ay, woes of Warwick, an oft fair assembly,
Pursued fearful my mind meh spake my life.

GLOUCESTER:
Harly, weep sin, at it is nearer to them;
And call thee mine:
To deep return faces and less to his eyes;
I speak 't up, when leave for sway again,
Our whole pregisidens upages him with the king's.
I'll tell wayer, every grievours his revenge
And Juliet by my fair promagation
Not kill me are upon your own is death?
And I hot spite the over-mouth to 
```

Well, there we have it. We could probably train our model further and make it produce more and more English like text, but let’s stop and think for a second, do we really need to generate our text character by character? We can actually make the process much easier for our model and simply split the text into individual words rather than individual characters. 
In my next post, I will present a way to generate our text using a word-by-word split rather than a character-by-character split.

<a href="/_notebooks/how-to-use-text-with-rnn.ipynb" download="download">Download as an .ipynb Jupyter Notebook</a>

