---
layout: post
title: Employing Recurrent Neural Networks for Text Generation - Part II
mathjax: true
image: writing-logo.png
---

In the [previous post](https://bshabashfd.github.io/2019/08/03/how-to-use-text-with-rnn.html) We've explored how we can use TensorFlow, Keras, and the Recurrent Neural Network (RNN) architecture to produce a system which can generate new Shakespearean prose for us. However, we ran into a problem... The network spends most of its time learning how to compose English words. It learns the form very quickly, but then we need many epochs to make the network spell correct words. In this tutorial we will explore how to solve that problem.


## Why are we having this issue?

In essence, the problem comes from the way we constructed our vocabulary. We split the entire text into individual characters and tried to predict the next character in the sequence. However, we can employ a more reasonable approach and split the data collection into words instead. This will allow the network to choose from already valid words. This seems easy enough, but there are in fact a few hurdles this approach will introduce. Let's work through each one as they come up:

## Phase 1: Import Keras and Obtain the data

We'll begin again by importing all required packages, and fixing the random seeds

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

```python
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print('length of text: {} chars'.format(len(text)))
print(text[:200])
```
```
length of text: 1115394 chars
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you
```

## Phase 2: Split the text into individual words
We're going to split our text into individual words this time, and let the network build text one word at a time.

Alright, let's split into individual words:

```python
words = text.split(' ')
print(words[:10])
```
```
['First', 'Citizen:\nBefore', 'we', 'proceed', 'any', 'further,', 'hear', 'me', 'speak.\n\nAll:\nSpeak,', 'speak.\n\nFirst']
```
**Ah ha!** we see our first problem... Some words contain a newline `\n` character (some even contain 2). This is an issue, we want to split words separated both by newlines and spaces. Let's fix that shall we?

```python
text_with_no_newlines = text.replace('\n\n', '\n').replace('\n', ' ')
print(text_with_no_newlines[:400])
```
```
First Citizen: Before we proceed any further, hear me speak. All: Speak, speak. First Citizen: You are all resolved rather to die than to famish? All: Resolved. resolved. First Citizen: First, you know Caius Marcius is chief enemy to the people. All: We know't, we know't. First Citizen: Let us kill him, and we'll have corn at our own price. Is't a verdict? All: No more talking on't; let it be done
```

Notice we replaced our double newlines with a single newlines, and then replaced all single newlines with a space. Now let's see what happens when we split.

```python
words = text_with_no_newlines.split(' ')
print(words[:10])
```
```
['First', 'Citizen:', 'Before', 'we', 'proceed', 'any', 'further,', 'hear', 'me', 'speak.']
```

Hmmmm... this is better, but notice some words have punctuation marks such as colons and periods. Let's see if we can find all punctuation marks in our text.

```python
all_chars = list(set(text))
for char in all_chars:
    if not (char.isalpha()):
        print(repr(char))
```
```
'&'
'$'
"'"
'.'
':'
' '
';'
','
'\n'
'!'
'3'
'-'
'?'
```
