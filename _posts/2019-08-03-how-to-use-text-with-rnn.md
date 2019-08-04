---
layout: post
title: Employing Recurrent Neural Networks for Text Generation
---

Recurrent Neural Networks (RNNs) are naturally designed to process sequences and extract patterns from them. Unlike Feed-Forward NNs which accept simply a collection of inputs as features, RNNs can pay attention to the order of the inputs coming in, and as such are often used to explore patterns through time or through any naturally occurring sequence.

In this post, we'll explore how to employ RNNs for text processing and generation, and we'll be using [TensorFlow](https://www.tensorflow.org/install) and its Keras API. This post is loosely based on the tutorial at https://www.tensorflow.org/beta/tutorials/text/text_generation

## Phase 1: Obtain the data

Lucky for us, TensorFlow comes shipped with an example text data. Let's import everything we need, and grab our text data:

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

print(text[:100])
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

We're going to use this text collection to train an RNN which can generate Shakespeare-like writing. We will give it a string of length n characters, and it will generate the next character for us. We see we have 65 unique characters, so this will be a 65 class classification problem. We will provide a single string of length n, and will process it to create an output of 1 character.

Simply speaking, we'd like our network to learn to print "o" when we provide the string "hell" (to make "hello") or "e" when we provide the string "goodby" (to make "goodbye").

let's define our string length to be 100

```python
seq_length = 100
```

# Phase 3: Pre-process our data:

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
seq_length = 100


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

Let’s take a look at out created datasets:
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
