---
layout: post
title: Employing Recurrent Neural Networks for Text Generation
---

Recurrent Neural Networks (RNNs) are naturally designed to process sequences and extract patterns from them. Unlike Feed-Forward NNs which accept simply a collection of inputs as features, RNNs can pay attention to the order of the inputs coming in, and as such are often used to explore patterns through time or through any naturally occuring sequence.

In this post, we'll explore how to employ RNNs for text processing and generation, and we'll be using [TensorFlow](https://www.tensorflow.org/install) and its Keras API. This post is loosly based on the tutorial at https://www.tensorflow.org/beta/tutorials/text/text_generation

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


